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

    def __init__(self):
        from torch_xcpu.ops_defs.moe_grouped_gemm import PortableBf16MoeGroupedGemm

        expert_map = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
        self.routed_experts = {
            str(i): SimpleNamespace(expert_map=expert_map) for i in self.layer_indices
        }
        self.backends = {}
        for i in self.layer_indices:

            def params(shape, value=i):
                return SimpleNamespace(
                    packed_weight=torch.full(shape, value, dtype=self.dtype),
                    packed_weight_scale=None,
                    bias=None,
                    scale_block_size=None,
                )

            self.backends[i] = SimpleNamespace(
                backend_type=PortableBf16MoeGroupedGemm,
                params=SimpleNamespace(
                    hidden=16,
                    intermediate=8,
                    experts=4,
                    gemm1=SimpleNamespace(params=params((4, 16, 16))),
                    gemm2=SimpleNamespace(params=params((4, 16, 8))),
                ),
            )

    def fused_moe_for_layer(self, layer_idx):
        return self.backends[layer_idx]


def test_expert_service_reuses_workspace_and_binds_each_layer(
    monkeypatch, make_af_session
):
    service = ExpertServiceV7(FakeModel(), make_af_session(ClusterType.MOE, max_rows=4))
    service.initialize()
    workspace = service._workspace
    calls = []

    def execute(*args):
        calls.append(args)
        args[-1].fill_(args[13])

    service._op = execute
    service.execute_model_pass()
    service.execute_model_pass()
    assert [args[13] for args in calls] == [1, 3, 1, 3]
    assert all(args[-1] is workspace for args in calls)
    for args in calls:
        assert (
            args[0]
            is service.model.backends[args[13]].params.gemm1.params.packed_weight
        )
        assert args[10:13] == (4, 6, 2)
    assert workspace.eq(3).all()


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
        graph.graph.eliminate_dead_code()
        assert (
            torch.ops.torch_xcpu.fused_af_f_moe_v7_PortableBf16MoeGroupedGemm.default
            in targets
        )
        assert any(
            node.target
            == torch.ops.torch_xcpu.fused_af_f_moe_v7_PortableBf16MoeGroupedGemm.default
            for node in graph.graph.nodes
        )
