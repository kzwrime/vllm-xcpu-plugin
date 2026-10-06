from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.af_ep.attn import runtime
from vllm_xcpu_plugin.af_ep.attn.client_v8 import ExpertsClientV8


def test_attention_runs_remote_layers_without_local_weights_or_double_reduce(
    monkeypatch, make_af_session_v8
):
    import torch_xcpu
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts

    dispatched = {}

    def dispatch(*args):
        hidden, ids, weights, layer_idx = args[3], args[4], args[5], args[9]
        assert ids.dtype == torch.int32 and ids.is_contiguous()
        assert weights.dtype == torch.float32 and weights.is_contiguous()
        dispatched[layer_idx] = hidden * weights.sum(dim=1, keepdim=True) * layer_idx

    def combine(*args):
        args[0].copy_(dispatched.pop(args[7]))

    monkeypatch.setattr(torch_xcpu.ops, "moe_af_dispatch_send_v8", dispatch)
    monkeypatch.setattr(torch_xcpu.ops, "moe_af_combine_recv_v8", combine)
    monkeypatch.setattr(
        moe_runner,
        "tensor_model_parallel_all_reduce",
        lambda _: pytest.fail("remote output was reduced twice"),
    )
    client = ExpertsClientV8(make_af_session_v8())
    client.initialize(16, 6, torch.bfloat16)
    monkeypatch.setattr(runtime, "get_remote_experts_client", lambda: client)

    outputs = []
    for layer_idx, rows in [(1, 2), (3, 5)]:
        method = XcpuRoutedExperts._get_quant_method(
            None, f"model.layers.{layer_idx}.mlp.experts", None, SimpleNamespace()
        )
        layer = torch.nn.Module()
        layer.global_num_experts, layer.local_num_experts = 8, 4
        method.create_weights(layer, 8, 16, 8, torch.bfloat16)
        layer.quant_method = method
        assert (
            XcpuRoutedExperts.load_weights(
                layer, iter([("0.gate_proj.weight_packed", torch.ones(1))])
            )
            == ()
        )
        method.process_weights_after_loading(layer)
        assert list(layer.parameters()) == [] and list(layer.buffers()) == []

        output = method.apply(
            layer,
            torch.ones(rows, 16, dtype=torch.bfloat16),
            torch.ones(6, rows, dtype=torch.bfloat16).t(),
            torch.arange(6, dtype=torch.int64).expand(rows, 6),
            None,
            None,
        )
        # Use vLLM's actual final-output handling to catch accidental double reduce.
        runner = SimpleNamespace(
            _quant_method=method,
            moe_config=SimpleNamespace(
                is_sequence_parallel=False,
                skip_final_all_reduce=False,
                tp_size=2,
                ep_size=2,
            ),
        )
        runner._fused_output_is_reduced = (
            moe_runner.MoERunner._fused_output_is_reduced.fget(runner)
        )
        outputs.append(
            moe_runner.MoERunner._maybe_reduce_final_output(runner, output, None)
        )

    assert not dispatched
    torch.testing.assert_close(outputs[0], torch.full((2, 16), 6, dtype=torch.bfloat16))
    torch.testing.assert_close(
        outputs[1], torch.full((5, 16), 18, dtype=torch.bfloat16)
    )


def test_attention_uses_remote_expert_partition_when_rank_counts_differ(
    monkeypatch, make_af_session_v8
):
    import torch_xcpu

    observed = {}

    def dispatch(*args):
        observed["num_experts"] = args[6]
        observed["num_local_experts"] = args[7]
        args[0].zero_()

    monkeypatch.setattr(torch_xcpu.ops, "moe_af_dispatch_send_v8", dispatch)
    monkeypatch.setattr(
        torch_xcpu.ops,
        "moe_af_combine_recv_v8",
        lambda output, *args: output.zero_(),
    )
    client = ExpertsClientV8(
        make_af_session_v8(num_attention_ranks=4, num_expert_ranks=1)
    )
    client.initialize(16, 6, torch.bfloat16)

    client.execute_layer(
        layer_idx=1,
        hidden_states=torch.ones(2, 16, dtype=torch.bfloat16),
        topk_weights=torch.ones(2, 6, dtype=torch.float32),
        topk_ids=torch.arange(6, dtype=torch.int64).expand(2, 6),
        num_experts=8,
        num_local_experts=2,
    )

    assert observed == {"num_experts": 8, "num_local_experts": 8}


@pytest.mark.parametrize(
    "rows,num_f,capacity", [(3, 2, 8), (128, 8, 256), (129, 8, 256)]
)
def test_attention_traces_dispatch_and_combine_with_checks_enabled(
    monkeypatch, make_af_session_v8, rows, num_f, capacity
):
    """Tracing must keep both endpoints without reading FakeTensor metadata."""
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch_xcpu.ops_defs import moe_af_v8

    monkeypatch.setattr(moe_af_v8, "_ENABLE_CHECKS", True)
    client = ExpertsClientV8(
        make_af_session_v8(num_expert_ranks=num_f, max_rows=capacity)
    )
    client.initialize(16, 6, torch.bfloat16)

    def remote(x, weights, ids):
        return client.execute_layer(
            layer_idx=1,
            hidden_states=x,
            topk_weights=weights,
            topk_ids=ids,
            num_experts=8,
            num_local_experts=4,
        )

    x = torch.ones(rows, 16, dtype=torch.bfloat16)
    weights = torch.ones(rows, 6, dtype=torch.float32)
    ids = torch.zeros(rows, 6, dtype=torch.int32)
    graph = make_fx(remote, tracing_mode="fake", _allow_non_fake_inputs=True)(
        x, weights, ids
    )
    endpoints = [
        node.target
        for node in graph.graph.nodes
        if node.op == "call_function" and "moe_af_" in str(node.target)
    ]
    assert endpoints == [
        torch.ops.torch_xcpu.moe_af_dispatch_send_v8_bf16.default,
        torch.ops.torch_xcpu.moe_af_combine_recv_v8_bf16.default,
    ]


def test_attention_reuses_reduced_send_workspace_across_pack_threshold(
    monkeypatch, make_af_session_v8
):
    import torch_xcpu

    buffers = []
    monkeypatch.setattr(
        torch_xcpu.ops, "moe_af_dispatch_send_v8", lambda *args: buffers.append(args[2])
    )
    monkeypatch.setattr(
        torch_xcpu.ops, "moe_af_combine_recv_v8", lambda output, *args: output.zero_()
    )
    client = ExpertsClientV8(make_af_session_v8(num_expert_ranks=8, max_rows=256))
    client.initialize(16, 6, torch.bfloat16)
    for rows in (128, 129, 256, 1):
        client.execute_layer(
            layer_idx=1,
            hidden_states=torch.ones(rows, 16, dtype=torch.bfloat16),
            topk_weights=torch.ones(rows, 6, dtype=torch.float32),
            topk_ids=torch.zeros(rows, 6, dtype=torch.int32),
            num_experts=8,
            num_local_experts=4,
        )
    assert all(buffer is buffers[0] for buffer in buffers)
    # 大批次上界为 6*256 行，覆盖小批次的 8*128 行；每条记录 8+8*6+16*2 字节。
    assert buffers[0].numel() == 1536 * 88


def test_v8_initialization_validates_protocol(monkeypatch, make_af_session_v8):
    from torch_xcpu.ops_defs import moe_af_v8

    monkeypatch.setattr(moe_af_v8, "_ENABLE_CHECKS", True)
    session = make_af_session_v8()
    moe_af_v8.moe_af_v8_initialize_check(
        8, 16, 6, torch.bfloat16, session.metadata, session.communicator_handle
    )
    old_metadata = session.metadata.clone()
    old_metadata[0] = 1
    with pytest.raises(RuntimeError):
        moe_af_v8.moe_af_v8_initialize_check(
            8, 16, 6, torch.bfloat16, old_metadata, session.communicator_handle
        )
