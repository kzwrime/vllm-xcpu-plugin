import pytest
import torch
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)

from vllm_xcpu_plugin.layers.fused_moe import (
    torch_all_to_all_single_prepare_finalize as ep,
)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mcpu"])
def test_prepare_counts_and_finalize_restores_rows(monkeypatch, empty, device):
    if device == "mcpu":
        import torch_mcpu  # noqa: F401
        import torch_xcpu

        torch_xcpu.initialize_runtime()
    monkeypatch.setattr(ep.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(ep.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        ep.dist, "all_to_all_single", lambda output, source, **kw: output.copy_(source)
    )
    prepare = ep.TorchAlltoallSinglePrepareAndFinalize(object(), 2, 1)
    ids = (
        torch.empty((0, 1), dtype=torch.int32)
        if empty
        else torch.tensor([[0], [1], [0]])
    )
    ids = ids.to(device)
    hidden = torch.ones((len(ids), 4)).to(device)
    weights = torch.ones(ids.shape).to(device)
    mapping = torch.tensor([1, 0], dtype=torch.int32).to(device)
    received, _, meta, received_ids, _ = prepare.prepare(
        hidden, weights, ids, 2, mapping, False, FUSED_MOE_UNQUANTIZED_CONFIG
    )
    assert torch.equal(received, hidden)
    assert torch.equal(received_ids, ids)
    assert meta.expert_num_tokens.dtype == torch.int32
    assert meta.expert_num_tokens.cpu().tolist() == ([0, 0] if empty else [1, 2])
    output = torch.empty_like(hidden)
    prepare.finalize(
        output, received, weights, ids, False, TopKWeightAndReduceDelegate()
    )
    assert torch.equal(output.cpu(), hidden.cpu())
