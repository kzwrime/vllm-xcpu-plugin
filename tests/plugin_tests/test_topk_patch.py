import torch
import torch_xcpu

from vllm_xcpu_plugin.topk_patch import _xcpu_topk_softmax


def test_xcpu_topk_softmax_forwards_is_padding(monkeypatch):
    captured = {}

    def fake_topk_softmax(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(torch_xcpu.ops, "topk_softmax", fake_topk_softmax)

    topk_weights = torch.empty(3, 2, dtype=torch.float32, device="cpu")
    topk_ids = torch.empty(3, 2, dtype=torch.int32, device="cpu")
    token_expert_indices = torch.empty(
        3, 2, dtype=torch.int32, device="cpu"
    )
    gating_output = torch.empty(3, 8, dtype=torch.float32, device="cpu")
    is_padding = torch.tensor([False, True, False], device="cpu")

    _xcpu_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=True,
        is_padding=is_padding,
    )

    args = captured["args"]
    assert len(args) == 5
    assert args[0] is topk_weights
    assert args[1] is topk_ids
    assert args[2] is token_expert_indices
    assert args[3] is gating_output
    assert args[4] is True
    assert captured["kwargs"]["is_padding"] is is_padding
