"""Checkpoint filtering for A ranks before model-specific weight loaders run."""

from dataclasses import replace

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

_FORMAT_PREFIX = "xcpu_af_"
_BASE_FORMATS = (
    "auto",
    "hf",
    "safetensors",
    "fastsafetensors",
    "instanttensor",
    "mistral",
    "pt",
    "npcache",
)


def configure_attention_model_loader(load_config: LoadConfig) -> None:
    """Select the AF loader without changing the checkpoint's storage format."""
    load_format = load_config.load_format
    if load_format == "dummy":
        return
    base_format = load_format.removeprefix(_FORMAT_PREFIX)
    if base_format not in _BASE_FORMATS:
        raise ValueError(
            f"AF-EP attention does not support load format {load_format!r}"
        )
    load_config.load_format = _FORMAT_PREFIX + base_format


class AttentionModelLoader(DefaultModelLoader):
    """Audit remote expert weights and pass local weights to the original model.

    Model loaders such as DeepseekV2Model.load_weights map every expert tensor
    to a local FusedMoE parameter. A owns none, so remote tensors must be
    consumed here before the model-specific loader sees them.
    """

    def __init__(self, load_config: LoadConfig):
        base_format = load_config.load_format.removeprefix(_FORMAT_PREFIX)
        if base_format not in _BASE_FORMATS:
            raise ValueError(f"unsupported AF-EP base load format: {base_format}")
        super().__init__(replace(load_config, load_format=base_format))

    def get_all_weights(self, model_config, model):
        from .remote_fused_moe import RemoteExpertsFusedMoEMethod

        remote_methods = {
            method.layer_name: method
            for module in model.modules()
            if isinstance(
                method := getattr(module, "quant_method", None),
                RemoteExpertsFusedMoEMethod,
            )
        }
        if remote_methods:
            # A owns no EP shard. An empty local set lets the safetensors
            # iterator skip every routed weight/packed tensor without reading
            # it; scales and other metadata still arrive and are audited.
            self.local_expert_ids = set()
        for name, weight in super().get_all_weights(model_config, model):
            prefix, separator, relative_name = name.partition(".experts.")
            method = remote_methods.get(prefix + ".experts") if separator else None
            if method is not None:
                method.skip_checkpoint_weights(((relative_name, weight),))
            else:
                yield name, weight


for _base_format in _BASE_FORMATS:
    register_model_loader(_FORMAT_PREFIX + _base_format)(AttentionModelLoader)
