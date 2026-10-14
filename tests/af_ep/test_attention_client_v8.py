from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.af_ep.attn import runtime
from vllm_xcpu_plugin.af_ep.attn.client_v8 import ExpertsClientV8


def test_attention_runs_remote_layers_without_local_weights_or_double_reduce(
    monkeypatch, make_af_session_v8
):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts

    monkeypatch.setattr(
        moe_runner,
        "tensor_model_parallel_all_reduce",
        lambda _: pytest.fail("remote output reduced twice"),
    )
    client = ExpertsClientV8(make_af_session_v8())
    monkeypatch.setattr(runtime, "get_remote_experts_client", lambda: client)
    methods = {
        i: XcpuRoutedExperts._get_quant_method(
            None,
            f"model.layers.{i}.mlp.experts",
            None,
            SimpleNamespace(moe_parallel_config=SimpleNamespace(sp_size=1)),
        )
        for i in (1, 3)
    }
    client.initialize(16, 6, torch.bfloat16)

    def execute(
        output,
        hidden,
        ids,
        weights,
        experts,
        capacity,
        layer_idx,
        metadata,
        comm,
        workspace,
    ):
        assert ids.dtype == torch.int32 and ids.is_contiguous()
        assert weights.dtype == torch.float32 and weights.is_contiguous()
        output.copy_(hidden * weights.sum(dim=1, keepdim=True) * layer_idx)

    client._op = execute
    outputs = []
    for layer_idx, rows in ((1, 2), (3, 5)):
        method = methods[layer_idx]
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
        assert not list(layer.parameters()) and not list(layer.buffers())
        output = method.apply(
            layer,
            torch.ones(rows, 16, dtype=torch.bfloat16),
            torch.ones(6, rows, dtype=torch.bfloat16).t(),
            torch.arange(6, dtype=torch.int64).expand(rows, 6),
            None,
            None,
        )
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
    torch.testing.assert_close(outputs[0], torch.full((2, 16), 6, dtype=torch.bfloat16))
    torch.testing.assert_close(
        outputs[1], torch.full((5, 16), 18, dtype=torch.bfloat16)
    )


def test_attention_uses_remote_expert_partition_when_rank_counts_differ(
    make_af_session_v8,
):
    client = ExpertsClientV8(
        make_af_session_v8(num_attention_ranks=4, num_expert_ranks=1)
    )
    client.initialize(16, 6, torch.bfloat16)
    observed = []

    def execute(*args):
        observed.append((args[4], int(args[7][4])))
        args[0].zero_()

    client._op = execute
    client.execute_layer(
        layer_idx=1,
        hidden_states=torch.ones(2, 16, dtype=torch.bfloat16),
        topk_weights=torch.ones(2, 6),
        topk_ids=torch.zeros(2, 6, dtype=torch.int32),
        num_experts=8,
        num_local_experts=2,
    )
    # Native A entry derives E=global/F from the remote topology, not A's shard.
    assert observed == [(8, 1)]


@pytest.mark.parametrize(
    "rows,num_f,capacity", [(3, 2, 8), (128, 8, 256), (129, 8, 256)]
)
def test_attention_traces_one_workspace_without_exposing_aliased_views(
    make_af_session_v8, rows, num_f, capacity
):
    from torch.fx.experimental.proxy_tensor import make_fx

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

    graph = make_fx(remote, tracing_mode="fake", _allow_non_fake_inputs=True)(
        torch.ones(rows, 16, dtype=torch.bfloat16),
        torch.ones(rows, 6),
        torch.zeros(rows, 6, dtype=torch.int32),
    )
    graph.graph.eliminate_dead_code()
    endpoints = [
        n.target
        for n in graph.graph.nodes
        if n.op == "call_function" and "fused_af_a_dispatch_combine" in str(n.target)
    ]
    assert endpoints == [
        torch.ops.torch_xcpu.fused_af_a_dispatch_combine_v8_bf16.default
    ]


def test_attention_reuses_workspace_across_pack_threshold(make_af_session_v8):
    client = ExpertsClientV8(make_af_session_v8(num_expert_ranks=8, max_rows=256))
    client.initialize(16, 6, torch.bfloat16)
    buffers = []

    def execute(*args):
        buffers.append(args[-1])
        args[0].zero_()

    client._op = execute
    for rows in (128, 129, 256, 1):
        client.execute_layer(
            layer_idx=1,
            hidden_states=torch.ones(rows, 16, dtype=torch.bfloat16),
            topk_weights=torch.ones(rows, 6),
            topk_ids=torch.zeros(rows, 6, dtype=torch.int32),
            num_experts=8,
            num_local_experts=4,
        )
    assert all(buffer is buffers[0] for buffer in buffers)
    align = lambda n: (n + 63) // 64 * 64
    send = 1536 * 88 + 0
    assert buffers[0].numel() == align(256 * 6 * 4) + align(8 * 4) + align(
        max(send, 256 * 16 * 4)
    )


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
