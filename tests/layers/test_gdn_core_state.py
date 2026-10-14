"""GDN state and output parity while one guardless AOT graph changes branches.

Run in a subprocess because AOT compiler and platform registration are global.
The reference is v0.31's Python scheduling with the same existing MCPU kernels.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "layout,packed,ratio,compile_core",
    [("SD", False, 1, True), ("SD", True, 3, True), ("SD", True, 3, False)],
)
def test_native_core_live_state(tmp_path, layout, packed, ratio, compile_core):
    env = dict(
        os.environ,
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path),
        OMP_NUM_THREADS="4",
        VLLM_SSM_CONV_STATE_LAYOUT=layout,
        TORCH_XCPU_ENABLE_CHECK="0",
    )
    result = subprocess.run(
        [
            sys.executable,
            __file__,
            str(int(packed)),
            str(ratio),
            str(int(compile_core)),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: " in result.stdout


def _exercise_native_core(packed_decode, head_ratio, compile_core):
    from types import MethodType, SimpleNamespace

    import torch
    import torch_mcpu  # noqa: F401
    import torch_xcpu
    from torch._inductor.utils import run_and_get_code
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    from vllm_xcpu_plugin.gdn_metadata_patch import XcpuGDNAttentionMetadata
    from vllm_xcpu_plugin.gdn_patch import maybe_patch_gdn_attention
    from vllm_xcpu_plugin.layers.qwen_gdn_linear_attn import (
        XcpuChunkGatedDeltaRule,
        XcpuQwenGatedDeltaNetAttention,
    )

    maybe_patch_gdn_attention()
    package = Path(torch_xcpu.__file__).parent
    os.environ["AOTI_EXTRA_CFLAGS"] = (
        f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    os.environ["AOTI_EXTRA_LDFLAGS"] = (
        f"-Wl,-rpath,{package} {next(package.glob('_C*.so'))}"
    )
    os.environ["TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES"] = "torch_xcpu"
    cfg = VllmConfig()
    compiled_core = None
    reference_core = gdn.QwenGatedDeltaNetAttention._forward_core
    chunk = object.__new__(XcpuChunkGatedDeltaRule)

    def native_pair(layer, alias, mixed, b, a, out):
        XcpuQwenGatedDeltaNetAttention._forward_core_native(layer, mixed, b, a, out)
        XcpuQwenGatedDeltaNetAttention._forward_core_native(alias, mixed, b, a, out)

    def device_int(data):
        return torch.tensor(data, dtype=torch.int32, device="mcpu")

    def query_starts(lens):
        return device_int([0] + list(torch.tensor(lens).cumsum(0).tolist()))

    def run_case(mode, tp, padding, fresh, head_ratio=1):
        nonlocal compiled_core
        torch.manual_seed(91465)
        heads, dim, pool = (24 if head_ratio == 3 else 16 // tp), 128, 24
        key_heads = heads // head_ratio
        width = (2 * key_heads + heads) * dim
        spec = mode in ("spec", "spec_mixed", "spec_zero")
        spec_lengths = [1, 1] if mode == "spec_zero" else [3, 2]
        pure_spec = mode in ("spec", "spec_zero")
        if mode == "empty":
            lengths, decodes, prefills = [], 0, 0
        elif mode in ("packed", "decode"):
            lengths, decodes, prefills = [1, 1], 2, 0
        elif mode == "prefill":
            lengths, decodes, prefills = [1, 65], 0, 2
        elif mode == "mixed":
            lengths, decodes, prefills = [1, 1, 65], 2, 1
        elif pure_spec:
            lengths, decodes, prefills = [], 0, 0
        else:
            lengths, decodes, prefills = [3, 1], 0, 2
        total = sum(lengths) + (sum(spec_lengths) if spec else 0)
        meta = XcpuGDNAttentionMetadata(
            num_prefills=prefills,
            num_prefill_tokens=sum(lengths[decodes:]),
            num_decodes=decodes,
            num_decode_tokens=decodes,
            num_spec_decodes=2 if spec else 0,
            num_spec_decode_tokens=sum(spec_lengths) if spec else 0,
            num_actual_tokens=total,
        )
        slots = device_int([17, 20, 22][: len(lengths)])
        meta.non_spec_state_indices_tensor = slots
        meta.non_spec_query_start_loc = query_starts(lengths)
        meta.has_initial_state = torch.tensor([not fresh] * len(lengths), device="mcpu")
        meta.prefill_state_indices = slots[decodes:]
        meta.prefill_query_start_loc = query_starts(lengths[decodes:])
        meta.prefill_has_initial_state = meta.has_initial_state[decodes:]
        if spec:
            meta.spec_sequence_masks = torch.tensor(
                [True, True] if pure_spec else [False, True, False, True], device="mcpu"
            )
            meta.spec_query_start_loc = query_starts(spec_lengths)
            meta.spec_state_indices_tensor = device_int([[2, 3, 4], [10, 11, 12]])
            meta.num_accepted_tokens = device_int(
                [3, 2] if mode == "spec_zero" else [1, 2]
            )
            meta.spec_token_indx = device_int(
                list(range(sum(spec_lengths))) if pure_spec else [3, 4, 5, 7, 8]
            ).long()
            meta.non_spec_token_indx = device_int(
                [] if pure_spec else [0, 1, 2, 6]
            ).long()
        meta.xcpu_core_counts = torch.tensor(
            [prefills, decodes, 2 if spec else 0, total, decodes], dtype=torch.int64
        )
        names = (
            "has_initial_state",
            "spec_query_start_loc",
            "non_spec_query_start_loc",
            "spec_state_indices_tensor",
            "non_spec_state_indices_tensor",
            "spec_token_indx",
            "non_spec_token_indx",
            "num_accepted_tokens",
            "prefill_query_start_loc",
            "prefill_state_indices",
            "prefill_has_initial_state",
        )
        meta.xcpu_core_tensors = tuple(
            getattr(meta, name).view(getattr(meta, name).shape)
            if getattr(meta, name) is not None
            else torch.empty(0, dtype=torch.int32, device="mcpu")
            for name in names
        )
        n = total + padding
        storage = (torch.randn(n, width + heads * dim, dtype=torch.bfloat16) * 0.1).to(
            "mcpu"
        )
        ba = (torch.randn(n, 2 * heads, dtype=torch.bfloat16) * 0.1).to("mcpu")
        b, a = ba.split(heads, dim=-1)
        conv_len = 8
        conv_shape = (
            (pool, width, conv_len)
            if gdn.is_conv_state_dim_first()
            else (pool, conv_len, width)
        )
        conv0 = (torch.randn(conv_shape, dtype=torch.bfloat16) * 0.05).to("mcpu")
        state0 = (torch.randn(pool, heads, dim, dim) * 0.05).to("mcpu")
        layer = SimpleNamespace(
            _xcpu_conv_state_dim_first=gdn.is_conv_state_dim_first(),
            prefix="layer",
            enable_packed_recurrent_decode=packed_decode,
            tp_size=tp,
            num_k_heads=key_heads * tp,
            num_v_heads=heads * tp,
            head_k_dim=dim,
            head_v_dim=dim,
            key_dim=key_heads * tp * dim,
            value_dim=heads * tp * dim,
            activation="silu",
            A_log=(torch.randn(heads) * 0.1).to("mcpu"),
            dt_bias=(torch.randn(heads, dtype=torch.bfloat16) * 0.1).to("mcpu"),
            conv1d=SimpleNamespace(
                weight=(torch.randn(width, 1, 4, dtype=torch.bfloat16) * 0.1).to(
                    "mcpu"
                ),
                bias=(torch.randn(width, dtype=torch.bfloat16) * 0.1).to("mcpu"),
            ),
            chunk_gated_delta_rule=chunk.forward_oot,
        )
        for name in ("rearrange_mixed_qkv", "_forward_core_decode_non_spec"):
            setattr(
                layer,
                name,
                MethodType(getattr(gdn.QwenGatedDeltaNetAttention, name), layer),
            )
        results = []
        for old in (True, False):
            conv_bytes = conv0[0].numel() * 2
            state_bytes = (heads + 1) * dim * dim * 4
            pages = torch.empty(
                pool, conv_bytes + state_bytes, dtype=torch.int8, device="mcpu"
            )
            conv_view = pages[:, :conv_bytes].view(torch.bfloat16).view(conv_shape)
            conv_view.copy_(conv0)
            state_storage = (
                pages[:, conv_bytes:]
                .view(torch.float32)
                .view(pool, heads + 1, dim, dim)
            )
            state_storage.fill_(123.0)
            state_view = state_storage[:, :heads]
            state_view.copy_(state0)
            layer.kv_cache = (conv_view, state_view)
            # Shared hybrid pages exclude row padding from the layer view.
            layer._xcpu_packed_kv_cache = (
                pages[:, : conv_bytes + heads * dim * dim * 4]
                .view(torch.bfloat16)
                .view(pool, 1, 1, -1)
            )
            layer._xcpu_conv_shape = tuple(conv_shape[1:])
            layer.rearrange_mixed_qkv = MethodType(
                gdn.QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer
            )
            alias = SimpleNamespace(**vars(layer))
            alias._xcpu_packed_kv_cache = layer._xcpu_packed_kv_cache.view(
                layer._xcpu_packed_kv_cache.shape
            )
            out = (
                torch.zeros(n, heads, dim, dtype=torch.bfloat16, device="mcpu")
                if old
                else torch.full(
                    (n, heads, dim), float("nan"), dtype=torch.bfloat16, device="mcpu"
                )
            )
            mixed = storage.clone()[:, :width]
            if old and mode == "empty":
                # The Python core assumes a nonempty scheduled batch. For an
                # empty native invocation, the contract is zero output and an
                # unchanged cache, not an invocation of that reference.
                results.append((out.cpu(), *(x.cpu() for x in layer.kv_cache)))
                continue
            with set_forward_context({"layer": meta}, cfg, num_tokens=total):
                core = (
                    reference_core
                    if old
                    else XcpuQwenGatedDeltaNetAttention._forward_core_native
                )
                if not old and compile_core:
                    if compiled_core is None:
                        for tensor in (mixed, b, a, out):
                            torch._dynamo.mark_dynamic(tensor, 0)
                        with (
                            torch._dynamo.config.patch(enable_aot_compile=True),
                            torch._inductor.config.patch(
                                cpp_wrapper=True,
                                enable_auto_functionalized_v2=True,
                                size_asserts=False,
                            ),
                        ):
                            wrapper = torch.compile(
                                native_pair, fullgraph=True, dynamic=False
                            )
                            compiled_core, code = run_and_get_code(
                                wrapper.aot_compile,
                                ((layer, alias, mixed, b, a, out), {}),
                            )
                        compiled_core.disable_guard_check()
                        source = "\n".join(code)
                        assert "aoti_torch_mcpu_qwen_gdn_attention_core_state" in source
                        assert "aoti_torch_proxy_executor_call_function" not in source
                        assert "cpp_fused" not in source
                    compiled_core(layer, alias, mixed, b, a, out)
                else:
                    core(layer, mixed, b, a, out)
                    core(alias, mixed, b, a, out)
            assert torch.all(state_storage[:, heads:].cpu() == 123.0)
            results.append((out.cpu(), *(x.cpu() for x in layer.kv_cache)))
        for expected, actual in zip(*results):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not torch.isnan(results[1][0]).any()
        assert torch.count_nonzero(results[1][0][total:]) == 0
        return {
            "mode": mode,
            "head_ratio": head_ratio,
            "tp": tp,
            "padding": padding,
            "fresh": fresh,
            "tokens": total,
            "gate_stride": list(a.stride()),
            "output_and_states_exact": True,
            "layout": gdn.is_conv_state_dim_first(),
        }

    count = 0
    with torch.inference_mode():
        for tp in (2,) if compile_core else (1, 2):
            for mode in (
                "prefill",
                "decode",
                "mixed",
                "spec",
                "spec_mixed",
                "spec_zero",
                "empty",
            ):
                for padding in (0, 3):
                    for fresh in (
                        (False, True)
                        if mode in ("prefill", "mixed", "spec_mixed")
                        else (False,)
                    ):
                        print(
                            run_case(mode, tp, padding, fresh, head_ratio), flush=True
                        )
                        count += 1
    print("PASS:", count, flush=True)


if __name__ == "__main__":
    _exercise_native_core(
        bool(int(sys.argv[1])), int(sys.argv[2]), bool(int(sys.argv[3]))
    )
