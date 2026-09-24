# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def _indexer_qk_forward(q, weights, k, weight, bias, positions, cos_sin, cache):
    from vllm_xcpu_plugin.layers.sparse_attn_indexer import _indexer_qk_cache

    return _indexer_qk_cache(
        q,
        weights,
        k,
        weight,
        bias,
        positions,
        cos_sin,
        cache,
        "indexer.k_cache",
        1e-6,
        0.125,
        0.25,
        False,
    )


def test_qk_aot_current_slots_and_reload(monkeypatch, tmp_path):
    from pathlib import Path

    import torch_mcpu  # noqa: F401
    import torch_xcpu
    import vllm.forward_context as ctx
    from torch._inductor.utils import run_and_get_code

    package = Path(torch_xcpu.__file__).parent
    library = next(package.glob("_C*.so"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv(
        "AOTI_EXTRA_CFLAGS", f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    monkeypatch.setenv("AOTI_EXTRA_LDFLAGS", f"-Wl,-rpath,{package} {library}")
    monkeypatch.setenv("TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES", "torch_xcpu")

    def mcpu_rand(*shape):
        return torch.randn(*shape).to("mcpu")

    weight, bias = mcpu_rand(128), mcpu_rand(128)
    cos_sin = mcpu_rand(37, 64).bfloat16()

    def inputs(rows, values):
        slots = torch.tensor(values, dtype=torch.int64, device="mcpu")
        monkeypatch.setattr(
            ctx,
            "_forward_context",
            ctx.ForwardContext(
                no_compile_layers={},
                attn_metadata={"indexer.k_cache": object()},
                slot_mapping={"indexer.k_cache": slots},
            ),
        )
        q = mcpu_rand(rows, 32, 128).bfloat16()
        kw = mcpu_rand(rows, 160).bfloat16()
        positions = (torch.arange(rows) % 37).to("mcpu")
        cache = torch.full((2, 64, 132), 0xA5, dtype=torch.uint8, device="mcpu")
        return (
            q,
            kw[:, 128:],
            kw[:, :128],
            weight,
            bias,
            positions,
            cos_sin,
            cache,
        ), slots

    args, slots = inputs(9, [0, 2, 65, -1])
    for t in (args[0], args[1], args[2], args[5], slots):
        torch._dynamo.mark_dynamic(t, 0)
    torch._dynamo.reset()
    with (
        torch._dynamo.config.patch(enable_aot_compile=True),
        torch._inductor.config.patch(
            cpp_wrapper=True, enable_auto_functionalized_v2=False
        ),
    ):
        wrapper = torch.compile(_indexer_qk_forward, fullgraph=True, dynamic=False)
        compiled, code = run_and_get_code(wrapper.aot_compile, (args, {}))
        code = "\n".join(code)
        (tmp_path / "qk_generated_code.txt").write_text(code)
        # Both routes call the C++ kernel; neither may hide a Python custom op.
        assert (
            "aoti_torch_mcpu_fused_indexer_qk_rope_quant_cache_out" in code
            or (
                "aoti_torch_call_dispatcher("
                '"torch_xcpu::fused_indexer_qk_rope_quant_cache_out"'
            )
            in code
        )
        assert "PyObject_CallObject" not in code
        assert "vllm_xcpu.indexer_qk_cache" not in code
        assert "vllm_xcpu::indexer_qk_cache" not in code
        artifact = tmp_path / "qk.aot"
        compiled.save_compiled_function(str(artifact))
        with artifact.open("rb") as handle:
            loaded = torch.compiler.load_compiled_function(
                handle, f_globals=_indexer_qk_forward.__globals__
            )
        for target in (compiled, loaded):
            target.disable_guard_check()
            for rows, values in (
                (9, [71, -1, 6, 9]),
                (9, [12, 80, -1, 22]),
                (13, [5, -1, 68, 21, 40, 71]),
                (5, [90, -1, 24]),
                (1, [100]),
                (7, []),
                (7, [-1, -1, -1]),
            ):
                args, slots = inputs(rows, values)
                q, weights, k, weight, bias, positions, cos_sin, cache = args
                expected_cache = cache.clone()
                expected_q, expected_w = torch_xcpu.ops.fused_indexer_q_rope_quant(
                    positions, q, cos_sin, weights, 0.125, 0.25, False, False
                )
                torch_xcpu.ops.fused_indexer_k_norm_rope_cache(
                    k,
                    weight,
                    bias,
                    positions,
                    cos_sin,
                    slots,
                    expected_cache,
                    1e-6,
                    False,
                )
                actual_q, actual_w = target(*args)
                for actual, expected in (
                    (actual_q, expected_q),
                    (actual_w, expected_w),
                    (cache, expected_cache),
                ):
                    torch.testing.assert_close(
                        actual.cpu(), expected.cpu(), rtol=0, atol=0
                    )
    torch._dynamo.reset()


def test_qk_profile_no_write_and_reject_capture(monkeypatch):
    import torch_mcpu  # noqa: F401
    import torch_xcpu
    import vllm.forward_context as ctx

    monkeypatch.setattr(
        ctx,
        "_forward_context",
        ctx.ForwardContext(
            no_compile_layers={},
            attn_metadata=None,
            slot_mapping={},
            skip_compiled=True,
        ),
    )
    q = torch.randn(3, 32, 128).bfloat16().to("mcpu")
    kw = torch.randn(3, 160).bfloat16().to("mcpu")
    positions = torch.arange(3).to("mcpu")
    cos_sin = torch.randn(3, 64).bfloat16().to("mcpu")
    cache = torch.full((1, 64, 132), 0xA5, dtype=torch.uint8, device="mcpu")
    before = cache.cpu().clone()
    args = (
        q,
        kw[:, 128:],
        kw[:, :128],
        torch.ones(128, device="mcpu"),
        torch.zeros(128, device="mcpu"),
        positions,
        cos_sin,
        cache,
    )
    expected = torch_xcpu.ops.fused_indexer_q_rope_quant(
        positions, q, cos_sin, kw[:, 128:], 0.125, 0.25, False, False
    )
    actual = _indexer_qk_forward(*args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(cache.cpu(), before, rtol=0, atol=0)
    torch._dynamo.reset()
    with pytest.raises(Exception, match="must bypass compilation"):
        torch.compile(_indexer_qk_forward, fullgraph=True, backend="eager")(*args)
    torch._dynamo.reset()
