"""A-rank weightless RoutedExperts method for AF-EP."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

    from .runtime import ExpertsClient

_LAYER_PREFIX = re.compile(
    r"^(?:model|language_model\.model)\.layers\."
    r"(?P<layer>\d+)\.mlp\.experts$"
)


class _RemoteKernelDescriptor:
    """vLLM capability adapter, not an executable FusedMoEKernel.

    MoERunner queries output_is_reduced() to avoid reducing routed output again
    after the client returns, and to reduce shared-expert output separately.
    Remote execution currently cannot overlap shared experts. The method below
    overrides kernel-dependent base properties; actual compute uses the client.
    """

    is_monolithic = False
    can_overlap_shared_experts = False

    @staticmethod
    def output_is_reduced() -> bool:
        return True


class RemoteExpertsFusedMoEMethod(FusedMoEMethodBase):
    """Keep routing on A, delegate routed compute, and allocate no expert weights."""

    def __init__(
        self,
        moe: FusedMoEConfig,
        *,
        layer_name: str,
        client: ExpertsClient,
    ) -> None:
        super().__init__(moe)
        match = _LAYER_PREFIX.fullmatch(layer_name)
        if match is None:
            raise ValueError(f"unsupported AF-EP routed layer prefix {layer_name!r}")
        self.layer_idx = int(match.group("layer"))
        self.layer_name = layer_name
        self.client = client
        self.moe_kernel = _RemoteKernelDescriptor()  # type: ignore[assignment]
        self._seen_checkpoint_names: set[str] = set()
        self._weight_loading_finalized = False

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    @property
    def supports_internal_mk(self) -> bool:
        return True

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return False

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Keep A free of routed-expert parameters; F loads its own weights."""
        del (
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            extra_weight_attrs,
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        del layer
        self._weight_loading_finalized = True

    def skip_checkpoint_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> tuple[str, ...]:
        """Consume and audit the iterator without storing or transmitting weights.

        F loads its checkpoint independently. Return no locally loaded names to
        vLLM, while checking duplicates across all loader chunks for this layer.
        """
        if self._weight_loading_finalized:
            raise RuntimeError("AF-EP routed weights were already finalized")
        for name, _ in weights:
            if name in self._seen_checkpoint_names:
                raise ValueError(
                    f"duplicate skipped expert weight {self.layer_name}.{name}"
                )
            self._seen_checkpoint_names.add(name)
        return ()

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        return self.client.execute_layer(
            layer_idx=self.layer_idx,
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=layer.global_num_experts,
            num_local_experts=layer.local_num_experts,
        )
