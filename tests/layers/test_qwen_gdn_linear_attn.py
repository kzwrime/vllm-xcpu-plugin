import sys
import types
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_xcpu_plugin.layers.qwen_gdn_linear_attn import (
    XcpuChunkGatedDeltaRule,
    XcpuQwenGatedDeltaNetAttention,
)


def test_native_core_rejects_unsupported_ds_layout() -> None:
    module = "vllm_xcpu_plugin.layers.qwen_gdn_linear_attn"
    layer = object.__new__(XcpuQwenGatedDeltaNetAttention)
    with (
        patch(f"{module}.QwenGatedDeltaNetAttention.__init__", return_value=None),
        patch(f"{module}.envs.VLLM_XCPU_GDN_COMPILE", True),
        patch(f"{module}.is_conv_state_dim_first", return_value=True),
        pytest.raises(ValueError, match="requires VLLM_SSM_CONV_STATE_LAYOUT=SD"),
    ):
        layer.__init__()


def test_native_core_binds_and_releases_shared_cache() -> None:
    from vllm.v1.worker.utils import clear_layer_kv_caches

    layer = object.__new__(XcpuQwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer._xcpu_native_core = True
    layer.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
    layer.cache_config = types.SimpleNamespace(cache_dtype="auto")
    layer.get_state_shape = lambda: ((3, 8), (2, 4, 4))
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    # Layer pages have row padding and share storage with full-attention views.
    backing = torch.zeros(2 * 4 * 256, dtype=torch.int8)
    for offset in (0, 4 * 256):
        raw = backing[offset : offset + 4 * 256].view(4, 1, 1, 256)
        layer.bind_kv_cache(raw)
        carrier = layer._xcpu_packed_kv_cache
        assert carrier.dtype == torch.bfloat16
        assert carrier.data_ptr() == raw.data_ptr()
        layer.kv_cache[0].fill_(2)
        layer.kv_cache[1].fill_(3)
        pages = carrier.view(torch.int8).view(4, 256)
        torch.testing.assert_close(
            pages[:, :48].view(torch.bfloat16),
            torch.full((4, 24), 2, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            pages[:, 48:176].view(torch.float32), torch.full((4, 32), 3.0)
        )
        assert not pages[:, 176:].count_nonzero()
        clear_layer_kv_caches([layer])
        assert layer._xcpu_packed_kv_cache is None
        assert layer.kv_cache == []


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "q": torch.empty(1, 2, 1, 4),
        "k": torch.empty(1, 2, 1, 4),
        "v": torch.empty(1, 2, 2, 3),
        "g": torch.empty(1, 2, 2),
        "beta": torch.empty(1, 2, 2),
        "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
    }


def test_dispatches_to_separated_kernel_without_state_pool() -> None:
    expected = (torch.empty(1), torch.empty(1))
    ops = types.SimpleNamespace(
        chunk_gated_delta_rule_separated=Mock(return_value=expected),
        chunk_gated_delta_rule_separated_custom_v2=Mock(),
    )
    torch_xcpu = types.SimpleNamespace(ops=ops)
    layer = object.__new__(XcpuChunkGatedDeltaRule)

    with patch.dict(sys.modules, {"torch_xcpu": torch_xcpu}):
        actual = layer.forward_oot(
            **_inputs(),
            initial_state=torch.empty(1, 2, 3, 4),
        )

    assert actual == expected
    ops.chunk_gated_delta_rule_separated.assert_called_once()
    ops.chunk_gated_delta_rule_separated_custom_v2.assert_not_called()


def test_dispatches_to_custom_kernel_with_state_pool() -> None:
    expected = torch.empty(1)
    ops = types.SimpleNamespace(
        chunk_gated_delta_rule_separated=Mock(),
        chunk_gated_delta_rule_separated_custom_v2=Mock(return_value=expected),
    )
    torch_xcpu = types.SimpleNamespace(ops=ops)
    layer = object.__new__(XcpuChunkGatedDeltaRule)

    with patch.dict(sys.modules, {"torch_xcpu": torch_xcpu}):
        actual = layer.forward_oot(
            **_inputs(),
            ssm_state=torch.empty(3, 2, 3, 4),
            ssm_state_indices=torch.tensor([1], dtype=torch.int32),
            has_initial_state=torch.ones(1, dtype=torch.bool),
        )

    assert actual == (expected, None)
    ops.chunk_gated_delta_rule_separated.assert_not_called()
    ops.chunk_gated_delta_rule_separated_custom_v2.assert_called_once()


def test_chunk_writes_supplied_output_without_touching_padding() -> None:
    output = torch.arange(12.0).view(1, 2, 2, 3)
    final_state = torch.empty(1, 2, 3, 4)
    ops = types.SimpleNamespace(
        chunk_gated_delta_rule_separated=Mock(return_value=(output, final_state)),
    )
    layer = object.__new__(XcpuChunkGatedDeltaRule)
    target = torch.full((4, 2, 3), -99.0)
    with patch.dict(sys.modules, {"torch_xcpu": types.SimpleNamespace(ops=ops)}):
        actual, state = layer.forward_oot(
            **_inputs(),
            initial_state=final_state,
            core_attn_out=target,
        )
    assert actual is output
    assert state is final_state
    torch.testing.assert_close(target[:2], output.squeeze(0))
    torch.testing.assert_close(target[2:], torch.full((2, 2, 3), -99.0))
