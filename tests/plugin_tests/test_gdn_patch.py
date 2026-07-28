from __future__ import annotations

import torch_xcpu

from vllm_xcpu_plugin.gdn_patch import _xcpu_causal_conv1d_fn


def test_causal_conv1d_fn_forwards_cache_all_contract(monkeypatch):
    recorded = {}

    def fake_causal_conv1d_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(
        torch_xcpu.ops,
        "causal_conv1d_fn",
        fake_causal_conv1d_fn,
    )

    values = {name: object() for name in (
        "x",
        "weight",
        "bias",
        "conv_states",
        "query_start_loc",
        "cache_indices",
        "has_initial_state",
        "block_idx_first_scheduled_token",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
        "num_computed_tokens",
        "metadata",
    )}
    result = _xcpu_causal_conv1d_fn(
        values["x"],
        values["weight"],
        values["bias"],
        values["conv_states"],
        values["query_start_loc"],
        cache_indices=values["cache_indices"],
        has_initial_state=values["has_initial_state"],
        block_idx_first_scheduled_token=values[
            "block_idx_first_scheduled_token"
        ],
        block_idx_last_scheduled_token=values[
            "block_idx_last_scheduled_token"
        ],
        initial_state_idx=values["initial_state_idx"],
        num_computed_tokens=values["num_computed_tokens"],
        block_size_to_align=8,
        metadata=values["metadata"],
        validate_data=True,
    )

    assert result is values["x"]
    for name, value in values.items():
        assert recorded[name] is value
    assert recorded["block_size_to_align"] == 8
    assert recorded["validate_data"] is True
