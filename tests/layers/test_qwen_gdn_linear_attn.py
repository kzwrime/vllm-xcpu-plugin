import sys
import types
from unittest.mock import Mock, patch

import torch

from vllm_xcpu_plugin.layers.qwen_gdn_linear_attn import (
    XcpuChunkGatedDeltaRule,
)


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
