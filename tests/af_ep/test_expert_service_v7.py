from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.af_ep.moe.service_v7 import ExpertServiceV7
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType


class FakeModel:
    layer_indices = (1, 3)
    hidden_size = 16
    intermediate_size = 8
    num_experts = 8
    top_k = 6
    hidden_act = "silu"
    ep_size = 2
    ep_rank = 0
    device = torch.device("cpu")
    dtype = torch.bfloat16
    routed_experts = {str(i): SimpleNamespace(expert_map=None) for i in layer_indices}

    def fused_moe_for_layer(self, layer_idx):
        return layer_idx


def test_expert_service_returns_computed_rows_for_each_moe_layer(
    monkeypatch, make_af_session
):
    import torch_xcpu

    sent = []

    def receive(hidden, ids, weights, valid, elements, offsets, layer_idx, *args):
        hidden.fill_(float("nan"))
        hidden[:layer_idx].fill_(layer_idx)
        ids.zero_()
        weights.fill_(1)
        valid.fill_(layer_idx)
        elements.copy_(torch.tensor([layer_idx * 16, 0]))
        offsets.copy_(torch.tensor([0, layer_idx * 16]))

    def compute(**kw):
        rows = int(kw["num_input_rows_valid"].item())
        kw["output"][:rows] = (
            kw["hidden_states"][:rows]
            * kw["topk_weights"][:rows].sum(dim=1, keepdim=True)
            * kw["backend"]
        )

    def send(output, elements, offsets, metadata, comm, capacity, topk, layer_idx):
        rows = int(elements.sum().item()) // 16
        sent.append((layer_idx, output[:rows].clone()))

    monkeypatch.setattr(torch_xcpu.ops, "moe_af_dispatch_recv_v7", receive)
    monkeypatch.setattr(torch_xcpu.ops, "fused_moe_compute", compute)
    monkeypatch.setattr(torch_xcpu.ops, "moe_af_combine_send_v7", send)
    service = ExpertServiceV7(FakeModel(), make_af_session(ClusterType.MOE, max_rows=4))
    service.initialize()

    service.execute_model_pass()

    assert [layer for layer, _ in sent] == [1, 3]
    torch.testing.assert_close(sent[0][1], torch.full((1, 16), 6, dtype=torch.bfloat16))
    torch.testing.assert_close(
        sent[1][1], torch.full((3, 16), 54, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_combine_send_survives_dead_code_elimination(monkeypatch, dtype):
    """The remote window write must survive even with no tensor outputs."""
    from torch._higher_order_ops.effects import _get_effect
    from torch._library.effects import EffectType
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch_xcpu.ops_defs import moe_af_v7

    monkeypatch.setattr(moe_af_v7, "_ENABLE_CHECKS", True)
    output = torch.empty(8, 16, dtype=dtype)
    elements = torch.empty(2, dtype=torch.int32)
    offsets = torch.empty_like(elements)
    metadata = torch.empty(8, dtype=torch.int64)
    comm = torch.empty(1, dtype=torch.int64)

    def send(output, elements, offsets, metadata, comm):
        moe_af_v7.moe_af_combine_send_v7(
            output, elements, offsets, metadata, comm, 4, 6, 1
        )

    graph = make_fx(send, tracing_mode="fake")(
        output, elements, offsets, metadata, comm
    )
    graph.graph.eliminate_dead_code()
    suffix = "bf16" if dtype == torch.bfloat16 else "fp32"
    endpoint = getattr(torch.ops.torch_xcpu, f"moe_af_combine_send_v7_{suffix}").default
    assert _get_effect(endpoint) == EffectType.ORDERED
    assert [
        node.target for node in graph.graph.nodes if node.op == "call_function"
    ] == [endpoint]


def test_expert_transaction_reuses_graph_across_many_layer_indices(
    monkeypatch, make_af_session
):
    """An int schema used to specialize every layer and hit the 8-graph limit."""
    from torch_xcpu.ops_defs import moe_af_v7

    monkeypatch.setattr(moe_af_v7, "_ENABLE_CHECKS", True)

    def compute(self, layer, ops):
        self._buffers.output.copy_(self._buffers.hidden_states)

    monkeypatch.setattr(ExpertServiceV7, "_compute", compute)
    service = ExpertServiceV7(FakeModel(), make_af_session(ClusterType.MOE, max_rows=4))
    service.initialize()
    graphs = []

    def capture(graph, inputs):
        graphs.append(graph)
        # Compile/guard regression only: never execute real MPI on CPU tensors.
        return lambda *args: None

    execute = torch.compile(
        service._execute_layer, backend=capture, fullgraph=True, dynamic=True
    )
    for index in range(12):
        execute(replace(service._layers[0], layer_idx=index))
    assert len(graphs) <= 3
    for graph in graphs:
        targets = [
            node.target for node in graph.graph.nodes if node.op == "call_function"
        ]
        assert torch.ops.torch_xcpu.moe_af_dispatch_recv_v7_bf16 in targets
        assert torch.ops.torch_xcpu.moe_af_combine_send_v7_bf16 in targets
