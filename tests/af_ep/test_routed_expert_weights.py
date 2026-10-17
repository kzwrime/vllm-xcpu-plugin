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


def test_expert_model_skips_checkpoint_mtp_layers():
    """GLM checkpoints include MTP experts outside the base model's F service."""
    from types import SimpleNamespace

    from vllm_xcpu_plugin.af_ep.moe.model import RoutedExpertsModel

    target = FakeTarget()
    target._weights_finalized = False
    target._vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_hidden_layers=3, num_nextn_predict_layers=1
            )
        )
    )
    target._process_weights_after_loading = lambda: None
    mtp_weight = ("model.layers.3.mlp.experts.0.down_proj.weight", torch.ones(1))
    loaded = RoutedExpertsModel.load_weights(
        target, iter([*routed_weights(), mtp_weight])
    )
    assert loaded == {"routed_experts.1.parameter", "routed_experts.2.parameter"}
    assert {layer for layer, _ in target.loaded} == {1, 2}
    assert mtp_weight[0] not in target.weight_audit.loaded_checkpoint_names
    # Only num_hidden_layers + i are MTP layers; later indices stay invalid.
    with pytest.raises(ValueError, match="non-MoE layer"):
        RoutedExpertsModel.load_weights(
            target,
            iter([("model.layers.4.mlp.experts.0.down_proj.weight", torch.ones(1))]),
        )


@pytest.mark.parametrize("load_format", ["auto", "safetensors", "pt"])
def test_attention_loader_filters_before_deepseek_loads_weights(
    tmp_path, default_vllm_config, load_format
):
    """GLM's DeepseekV2 loader must not look up absent A-side expert parameters."""
    from types import SimpleNamespace

    from safetensors.torch import save_file
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    from vllm_xcpu_plugin.af_ep.attn.model_loader import (
        configure_attention_model_loader,
    )
    from vllm_xcpu_plugin.af_ep.attn.remote_fused_moe import (
        RemoteExpertsFusedMoEMethod,
    )

    method = RemoteExpertsFusedMoEMethod(
        SimpleNamespace(moe_parallel_config=SimpleNamespace(sp_size=1)),
        layer_name="model.layers.0.mlp.experts",
        client=SimpleNamespace(register_layer_capacity=lambda _: None),
    )
    model = DeepseekV2Model.__new__(DeepseekV2Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=4, n_shared_experts=1)
    model.use_mha = False
    model.num_redundant_experts = 0
    model.is_fused_shared_expert_enabled = False
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    mlp = model.layers[0].mlp = torch.nn.Module()
    mlp.experts = torch.nn.Module()
    mlp.experts.quant_method = method
    mlp.shared_experts = torch.nn.Module()
    mlp.shared_experts.down_proj = torch.nn.Linear(2, 2, bias=False)
    shared_name = "layers.0.mlp.shared_experts.down_proj.weight"
    expert_weights = {
        f"model.layers.0.mlp.experts.{expert}.{projection}.{suffix}": torch.ones(1)
        for expert in (0, 1)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale", "weight_packed")
    }
    scale_names = {name for name in expert_weights if name.endswith("_scale")}
    checkpoint = {
        **expert_weights,
        f"model.{shared_name}": torch.full((2, 2), 3.0),
    }
    if load_format == "pt":
        torch.save(checkpoint, tmp_path / "model.pt")
    else:
        save_file(checkpoint, tmp_path / "model.safetensors")

    class CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def load_weights(self, weights):
            loaded = self.model.load_weights(
                (name.removeprefix("model."), weight) for name, weight in weights
            )
            return {f"model.{name}" for name in loaded}

    load_config = LoadConfig(load_format=load_format, use_tqdm_on_load=False)
    configure_attention_model_loader(load_config)
    configure_attention_model_loader(load_config)  # idempotent
    loader = get_model_loader(load_config)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
        is_moe=True,
        multimodal_config=None,
    )
    causal_model = CausalLM()
    loader.load_weights(causal_model, model_config)

    assert load_config.load_format == f"xcpu_af_{load_format}"
    assert loader.load_config.load_format == load_format
    # safetensors skips remote weight/packed tensors before reading them;
    # pt has no lazy filter and audits every remote tensor.
    expected_seen = scale_names if load_format != "pt" else set(expert_weights)
    assert {
        f"model.layers.0.mlp.experts.{name}" for name in method._seen_checkpoint_names
    } == expected_seen
    assert not list(mlp.experts.parameters())
    torch.testing.assert_close(
        mlp.shared_experts.down_proj.weight, torch.full((2, 2), 3.0)
    )
    with pytest.raises(ValueError, match="duplicate skipped expert weight"):
        loader.load_weights(causal_model, model_config)


@pytest.mark.parametrize(
    "af_enabled,enable_ep,load_format,expected",
    [
        (True, True, "auto", "xcpu_af_auto"),
        (False, True, "auto", "auto"),
        (True, False, "auto", "auto"),  # F service does not enable vLLM EP
        (True, True, "dummy", "dummy"),
    ],
)
def test_platform_selects_attention_loader_only_for_afd(
    monkeypatch, af_enabled, enable_ep, load_format, expected
):
    from vllm.config import LoadConfig

    import vllm_xcpu_plugin.envs as envs_xcpu
    from vllm_xcpu_plugin.af_ep.attn.model_loader import (
        configure_attention_model_loader,
    )

    monkeypatch.setenv("VLLM_XCPU_ENABLE_AF_EP", str(int(af_enabled)))
    load_config = LoadConfig(load_format=load_format)
    # Mirror McpuPlatform.check_and_update_config's AF-EP branch.
    if envs_xcpu.VLLM_XCPU_ENABLE_AF_EP and enable_ep:
        configure_attention_model_loader(load_config)
    assert load_config.load_format == expected


def test_expert_loader_reads_only_local_expert_weights(tmp_path, default_vllm_config):
    """F skips other ranks' routed weight tensors before reading them."""
    from types import SimpleNamespace

    from safetensors.torch import save_file
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader

    from vllm_xcpu_plugin.af_ep.moe.runner import restrict_loader_to_local_experts

    checkpoint = {
        f"model.layers.1.mlp.experts.{expert}.down_proj.{suffix}": torch.ones(1)
        for expert in range(4)
        for suffix in ("weight", "weight_scale")
    }
    checkpoint["model.layers.1.mlp.gate.weight"] = torch.ones(1)
    save_file(checkpoint, tmp_path / "model.safetensors")

    class Model(torch.nn.Module):
        seen: list[str] = []

        def load_weights(self, weights):
            self.seen = [name for name, _ in weights]
            return set()

    loader = get_model_loader(LoadConfig(load_format="auto", use_tqdm_on_load=False))
    restrict_loader_to_local_experts(loader, (2, 3))
    model = Model()
    loader.load_weights(
        model,
        SimpleNamespace(
            model=str(tmp_path),
            revision=None,
            quantization=None,
            is_moe=True,
            multimodal_config=None,
        ),
    )
    assert set(model.seen) == {
        "model.layers.1.mlp.gate.weight",
        *(f"model.layers.1.mlp.experts.{e}.down_proj.weight" for e in (2, 3)),
        *(f"model.layers.1.mlp.experts.{e}.down_proj.weight_scale" for e in range(4)),
    }
