# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _indexer_cache_forward(k, weight, bias, positions, cos_sin, cache):
    from vllm_xcpu_plugin.layers.sparse_attn_indexer import _indexer_k_cache

    _indexer_k_cache(
        k, weight, bias, positions, cos_sin, cache, "indexer.k_cache", 1e-6, False
    )
    return cache


def test_indexer_cache_aot_reads_current_slot_mapping(monkeypatch, tmp_path):
    from pathlib import Path

    import torch_mcpu  # noqa: F401
    import torch_xcpu
    import vllm.forward_context as context_module
    from torch._inductor.utils import run_and_get_code

    package = Path(torch_xcpu.__file__).parent
    library = next(package.glob("_C*.so"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv(
        "AOTI_EXTRA_CFLAGS", f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    monkeypatch.setenv("AOTI_EXTRA_LDFLAGS", f"-Wl,-rpath,{package} {library}")
    monkeypatch.setenv("TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES", "torch_xcpu")

    def inputs(rows, values):
        slots = torch.tensor(values, dtype=torch.int64, device="mcpu")
        monkeypatch.setattr(
            context_module,
            "_forward_context",
            context_module.ForwardContext(
                no_compile_layers={},
                attn_metadata={"indexer.k_cache": object()},
                slot_mapping={"indexer.k_cache": slots},
            ),
        )
        k = torch.randn(rows, 160, device="cpu").bfloat16().to("mcpu")[:, :128]
        positions = (torch.arange(rows, device="cpu") % 37).to("mcpu")
        cache = torch.full((2, 64, 132), 0xA5, dtype=torch.uint8, device="mcpu")
        return (k, weight, bias, positions, cos_sin, cache), slots

    weight = torch.randn(128, device="cpu").to("mcpu")
    bias = torch.randn(128, device="cpu").to("mcpu")
    cos_sin = torch.randn(37, 64, device="cpu").bfloat16().to("mcpu")
    args, slots = inputs(9, [0, 2, 65, -1])
    for tensor in (args[0], args[3], slots):
        torch._dynamo.mark_dynamic(tensor, 0)
    torch._dynamo.reset()
    with (
        torch._dynamo.config.patch(enable_aot_compile=True),
        torch._inductor.config.patch(
            cpp_wrapper=True, enable_auto_functionalized_v2=False
        ),
    ):
        wrapper = torch.compile(_indexer_cache_forward, fullgraph=True, dynamic=False)
        compiled, code = run_and_get_code(wrapper.aot_compile, (args, {}))
        code = "\n".join(code)
        assert "aoti_torch_mcpu_fused_indexer_k_norm_rope_cache" in code
        assert "PyObject_CallObject" not in code
        assert "vllm_xcpu.indexer_k_cache" not in code
        artifact = tmp_path / "indexer.aot"
        compiled.save_compiled_function(str(artifact))
        with artifact.open("rb") as handle:
            loaded = torch.compiler.load_compiled_function(
                handle, f_globals=_indexer_cache_forward.__globals__
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
                expected = args[-1].clone()
                torch_xcpu.ops.fused_indexer_k_norm_rope_cache(
                    *args[:5], slots, expected, 1e-6, False
                )
                target(*args)
                torch.testing.assert_close(
                    args[-1].cpu(), expected.cpu(), rtol=0, atol=0
                )
    torch._dynamo.reset()


def test_indexer_cache_profile_skips_write_and_rejects_capture(monkeypatch):
    import vllm.forward_context as context_module

    monkeypatch.setattr(
        context_module,
        "_forward_context",
        context_module.ForwardContext(
            no_compile_layers={},
            attn_metadata=None,
            slot_mapping={},
            skip_compiled=True,
        ),
    )
    tensor = torch.ones(3, device="cpu")
    _indexer_cache_forward(tensor, tensor, tensor, tensor, tensor, tensor)
    torch.testing.assert_close(tensor, torch.ones_like(tensor))
    torch._dynamo.reset()
    compiled = torch.compile(_indexer_cache_forward, fullgraph=True, backend="eager")
    with pytest.raises(Exception, match="must bypass compilation"):
        compiled(tensor, tensor, tensor, tensor, tensor, tensor)
    torch._dynamo.reset()


def test_ir_priority_uses_functional_kernel_for_compile():
    from types import SimpleNamespace

    from vllm.config import CompilationMode

    from vllm_xcpu_plugin.platform import McpuPlatform

    McpuPlatform.import_ir_kernels()
    eager_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    compile_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.DYNAMO_TRACE_ONCE)
    )

    eager_priority = McpuPlatform.get_default_ir_op_priority(eager_config)
    assert eager_priority.rms_norm == ["torch_xcpu", "native"]
    assert eager_priority.fused_add_rms_norm == [
        "torch_xcpu_inplace",
        "torch_xcpu",
        "native",
    ]
    compile_priority = McpuPlatform.get_default_ir_op_priority(compile_config)
    assert compile_priority.rms_norm == ["torch_xcpu", "native"]
    assert compile_priority.fused_add_rms_norm == [
        "torch_xcpu",
        "native",
    ]


def test_compile_config_disables_inductor_fusions():
    from types import SimpleNamespace

    from vllm.config import CompilationMode

    from vllm_xcpu_plugin.platform import McpuPlatform

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=[],
            backend="",
            inductor_compile_config={},
        ),
        parallel_config=SimpleNamespace(worker_cls=None),
        cache_config=SimpleNamespace(
            user_specified_block_size=True,
            block_size=256,
        ),
        model_config=None,
    )

    McpuPlatform.check_and_update_config(config)

    compile_config = config.compilation_config.inductor_compile_config
    assert compile_config["epilogue_fusion"] is False
    assert compile_config["pattern_matcher"] is False
    assert compile_config["combo_kernels"] is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_norm_split_views_eager_and_compile(dtype, monkeypatch):
    """SP split views must select the fused provider without input copies."""
    from pathlib import Path

    import torch_mcpu  # noqa: F401
    import torch_xcpu
    from torch._inductor.utils import run_and_get_code
    from vllm import ir
    from vllm.ir.op import enable_torch_wrap

    from vllm_xcpu_plugin.platform import McpuPlatform

    McpuPlatform.import_ir_kernels()
    package = Path(torch_xcpu.__file__).parent
    library = next(package.glob("_C*.so"))
    monkeypatch.setenv(
        "AOTI_EXTRA_CFLAGS", f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    monkeypatch.setenv("AOTI_EXTRA_LDFLAGS", f"-Wl,-rpath,{package} {library}")
    monkeypatch.setenv("TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES", "torch_xcpu")

    def fn(packed, weight):
        x, residual = packed.split(64, dim=-1)
        return ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, 1e-6)

    weight = torch.ones(64, dtype=dtype, device="mcpu")
    op = ir.ops.fused_add_rms_norm
    torch._dynamo.reset()
    with enable_torch_wrap(False), op.set_priority(["torch_xcpu", "native"]):
        compiled = torch.compile(
            fn, fullgraph=True, dynamic=True, options={"cpp_wrapper": True}
        )
        for rows in (3, 7):
            cpu = torch.randn(rows, 128, dtype=dtype, device="cpu")
            packed = cpu.to("mcpu")
            x, residual = packed.split(64, dim=-1)
            assert op.dispatch(x, residual, weight, 1e-6).provider == "torch_xcpu"
            summed = cpu[:, :64].float() + cpu[:, 64:].float()
            expected = (
                summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            (out, residual_out), code = run_and_get_code(compiled, packed, weight)
            torch.testing.assert_close(out.cpu(), expected, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(
                residual_out.cpu(), summed.to(dtype), atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(packed.cpu(), cpu)
            if rows == 3:
                code = "\n".join(code)
                assert "aoti_torch_mcpu_fused_add_rms_norm_out" in code
                assert "torch.ops.aten.mean.dim" not in code
                assert "aoti_torch_mcpu_pow_Tensor_Scalar" not in code
                assert "aten::clone" not in code

    with (
        enable_torch_wrap(False),
        op.set_priority(["torch_xcpu_inplace", "torch_xcpu", "native"]),
    ):
        assert op.dispatch(x, residual, weight, 1e-6).provider == "torch_xcpu_inplace"
        out, residual_out = fn(packed, weight)
        assert out.data_ptr() == x.data_ptr()
        assert residual_out.data_ptr() == residual.data_ptr()
        torch.testing.assert_close(out.cpu(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            residual_out.cpu(), summed.to(dtype), atol=2e-2, rtol=2e-2
        )
    torch._dynamo.reset()
