from types import SimpleNamespace

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
