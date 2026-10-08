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


def test_python_forward_core_keeps_spec_gates_with_their_tokens(monkeypatch):
    """A same-length prefill preceding a spec row must not supply its gates."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cpu")

    meta = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=6,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=6,
        num_actual_tokens=12,
        has_initial_state=tensor([False], torch.bool),
        spec_query_start_loc=tensor([0, 6]),
        non_spec_query_start_loc=tensor([0, 6]),
        spec_state_indices_tensor=tensor([[1, 2, 3, 4, 5, 6]]),
        non_spec_state_indices_tensor=tensor([7]),
        spec_sequence_masks=tensor([False, True], torch.bool),
        spec_token_indx=torch.arange(6, 12, device="cpu"),
        non_spec_token_indx=torch.arange(6, device="cpu"),
        num_accepted_tokens=tensor([1]),
        prefill_query_start_loc=tensor([0, 6]),
        prefill_state_indices=tensor([7]),
        prefill_has_initial_state=tensor([False], torch.bool),
    )
    layer = types.SimpleNamespace(
        prefix="test",
        enable_packed_recurrent_decode=False,
        kv_cache=(
            torch.zeros(8, 8, 3, device="cpu"),
            torch.zeros(8, 1, 1, 1, device="cpu"),
        ),
        conv1d=types.SimpleNamespace(
            weight=torch.ones(3, 1, 4, device="cpu"), bias=None
        ),
        activation=None,
        A_log=torch.zeros(1, device="cpu"),
        dt_bias=torch.zeros(1, device="cpu"),
        num_k_heads=1,
        tp_size=1,
        head_k_dim=1,
        head_v_dim=1,
        enable_custom_prefill=True,
        rearrange_mixed_qkv=lambda x: (x[:, :1].view(1, -1, 1, 1),) * 3,
        chunk_gated_delta_rule=lambda **kw: (kw["q"], None),
    )
    ctx = types.SimpleNamespace(attn_metadata={"test": meta})
    monkeypatch.setattr(gdn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda x, *args, **kw: x)
    monkeypatch.setattr(gdn, "causal_conv1d_update", lambda x, *args, **kw: x)
    monkeypatch.setattr(
        gdn,
        "fused_post_conv_prep",
        lambda **kw: (
            *(kw["conv_output"][:, :1].view(-1, 1, 1),) * 3,
            kw["a"],
            kw["b"],
        ),
    )
    gates = {}

    def recurrent(**kw):
        gates.update(a=kw["a"], b=kw["b"])
        return kw["q"], None

    monkeypatch.setattr(gdn, "fused_sigmoid_gating_delta_rule_update", recurrent)
    a = torch.arange(12, device="cpu", dtype=torch.float32).view(12, 1)
    b = a + 20
    gdn.QwenGatedDeltaNetAttention._forward_core(
        layer,
        torch.zeros(12, 3, device="cpu"),
        b,
        a,
        torch.empty(12, 1, 1, device="cpu"),
    )
    torch.testing.assert_close(gates["a"], a[6:])
    torch.testing.assert_close(gates["b"], b[6:])
