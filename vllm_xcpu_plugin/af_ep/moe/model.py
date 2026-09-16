"""F-rank routed-experts-only model for AF-EP."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

import torch

from .weights import RoutedExpertsWeightLoader, WeightLoadAudit

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class RoutedExpertsModel(torch.nn.Module):
    """Own only local routed experts; never create gate, shared FFN, or attention."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        ep_size: int,
        ep_rank: int,
        max_num_tokens: int,
        device: torch.device | str = "mcpu",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        if str(model_config.dtype).lower() not in {
            "bf16",
            "bfloat16",
            "torch.bfloat16",
        }:
            raise ValueError("AF-EP expert service supports only BF16")
        if ep_size <= 1 or not 0 <= ep_rank < ep_size:
            raise ValueError("invalid F-side EP topology")
        num_experts = getattr(config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(config, "n_routed_experts", None)
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("AF-EP requires a routed MoE model")
        if num_experts % ep_size != 0:
            raise ValueError("AF-EP requires equal routed-expert shards on all F ranks")
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be positive")

        self.num_layers = config.num_hidden_layers
        self.layer_indices = self._moe_layer_indices(config)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_act = config.hidden_act
        self.dtype = model_config.dtype
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.max_num_tokens = max_num_tokens
        self.device = torch.device(device)
        self._vllm_config = vllm_config
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            self.routed_experts = torch.nn.ModuleDict(
                {
                    str(layer_idx): self._build_routed_experts(layer_idx)
                    for layer_idx in self.layer_indices
                }
            )
        self._weights_finalized = False
        self.weight_audit: WeightLoadAudit | None = None

    def local_expert_ids(self) -> tuple[int, ...]:
        local_count = self.num_experts // self.ep_size
        start = self.ep_rank * local_count
        return tuple(range(start, start + local_count))

    def load_routed_weight(
        self,
        layer_idx: int,
        relative_name: str,
        weight: torch.Tensor,
    ) -> Collection[str]:
        return tuple(
            self.routed_experts[str(layer_idx)].load_weights([(relative_name, weight)])
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        if self._weights_finalized:
            raise RuntimeError("AF-EP F-side weights cannot be hot reloaded")
        audit = RoutedExpertsWeightLoader(self).load(weights)
        self._process_weights_after_loading()
        self.weight_audit = audit
        return set(audit.loaded_parameter_names)

    def process_dummy_weights_after_loading(self) -> None:
        if self._weights_finalized:
            raise RuntimeError("AF-EP F-side weights cannot be hot reloaded")
        self._process_weights_after_loading()

    def _process_weights_after_loading(self) -> None:
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self._vllm_config):
            for layer in self.routed_experts.values():
                layer.quant_method.process_weights_after_loading(layer)
        self._weights_finalized = True

    def fused_moe_for_layer(self, layer_idx: int):
        if not self._weights_finalized:
            raise RuntimeError("expert weights have not been finalized")
        layer = self.routed_experts[str(layer_idx)]
        fused_moe = getattr(layer, "_xcpu_fused_moe", None)
        if fused_moe is None:
            raise RuntimeError("XCPU fused-MoE kernel was not installed")
        return fused_moe

    def _build_routed_experts(self, layer_idx: int):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            ExpertMapManager,
        )

        from vllm_xcpu_plugin.layers.fused_moe.routed_experts import (
            XcpuRoutedExperts,
        )

        parallel = FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=self.ep_size,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=self.ep_rank,
            sp_size=1,
            use_ep=True,
            all2all_backend="mpi_alltoallv_v7",
            enable_eplb=False,
        )
        expert_map_manager = ExpertMapManager(
            max_num_batched_tokens=self.max_num_tokens,
            top_k=self.top_k,
            global_num_experts=self.num_experts,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
            num_fused_shared_experts=0,
            rocm_aiter_enabled=False,
        )
        moe_config = FusedMoEConfig(
            num_experts=self.num_experts,
            experts_per_token=self.top_k,
            hidden_dim=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_local_experts=expert_map_manager.local_num_experts,
            num_logical_experts=self.num_experts,
            activation=MoEActivation.from_str(self.hidden_act),
            device=self.device,
            routing_method=RoutingMethodType.Default,
            moe_parallel_config=parallel,
            in_dtype=self.dtype,
            max_num_tokens=self.max_num_tokens,
        )
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        return XcpuRoutedExperts(
            layer_name=prefix,
            params_dtype=self.dtype,
            moe_config=moe_config,
            quant_config=self._vllm_config.quant_config,
            expert_map_manager=expert_map_manager,
        )

    @staticmethod
    def _moe_layer_indices(config) -> tuple[int, ...]:
        num_layers = int(config.num_hidden_layers)
        mlp_only_layers = frozenset(getattr(config, "mlp_only_layers", ()))
        decoder_sparse_step = getattr(config, "decoder_sparse_step", None)
        if decoder_sparse_step is not None:
            assert decoder_sparse_step > 0
            layers = tuple(
                layer_idx
                for layer_idx in range(num_layers)
                if layer_idx not in mlp_only_layers
                and (layer_idx + 1) % decoder_sparse_step == 0
            )
        else:
            first_moe_layer = int(getattr(config, "first_k_dense_replace", 0))
            moe_layer_freq = int(getattr(config, "moe_layer_freq", 1))
            assert 0 <= first_moe_layer < num_layers and moe_layer_freq > 0
            layers = tuple(
                layer_idx
                for layer_idx in range(first_moe_layer, num_layers)
                if layer_idx % moe_layer_freq == 0
            )
        if not layers:
            raise ValueError("AF-EP model has no routed MoE layers")
        return layers
