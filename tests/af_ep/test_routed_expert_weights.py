from dataclasses import dataclass, field

import pytest
import torch

from vllm_xcpu_plugin.af_ep.moe.weights import RoutedExpertsWeightLoader


@dataclass
class FakeTarget:
    layer_indices: tuple[int, ...] = (1, 2)
    num_experts: int = 4
    ep_size: int = 2
    loaded: list[tuple[int, str]] = field(default_factory=list)

    def local_expert_ids(self):
        return (2, 3)

    def load_routed_weight(self, layer_idx, relative_name, weight):
        self.loaded.append((layer_idx, relative_name))
        return ("parameter",)


def routed_weights():
    for layer_idx in (1, 2):
        for expert_id in range(4):
            for projection in ("gate_proj", "down_proj", "up_proj"):
                for suffix in ("weight", "weight_packed", "weight_scale_inv"):
                    name = (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_id}."
                        f"{projection}.{suffix}"
                    )
                    yield name, torch.empty(1)


def test_expert_loading_filters_ep_shard_and_preserves_quantized_tensor_names():
    target = FakeTarget()
    audit = RoutedExpertsWeightLoader(target).load([
        ("model.embed_tokens.weight", torch.empty(1)),
        ("model.layers.1.mlp.gate.weight", torch.empty(1)),
        ("model.layers.1.mlp.shared_expert.gate_proj.weight", torch.empty(1)),
        *routed_weights(),
    ])

    expected = {
        (layer, f"{expert}.{projection}.{suffix}")
        for layer in (1, 2)
        for expert in (2, 3)
        for projection in ("gate_proj", "down_proj", "up_proj")
        for suffix in ("weight", "weight_packed", "weight_scale_inv")
    }
    assert len(target.loaded) == len(expected) == audit.loaded_count
    assert set(target.loaded) == expected
    assert len(audit.remote_checkpoint_names) == len(expected)
    assert audit.skipped_non_routed_weights == 3


def test_missing_expert_layer_weights_reject_startup():
    weights = [item for item in routed_weights() if ".layers.2." not in item[0]]
    with pytest.raises(ValueError, match=r"layers \[2\]"):
        RoutedExpertsWeightLoader(FakeTarget()).load(weights)
