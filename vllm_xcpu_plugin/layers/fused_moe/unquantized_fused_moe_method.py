"""Unquantized method mapped to the unified XCPU grouped-GEMM pipeline."""

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from .setup import install_fused_moe, reject_fused_moe_hot_reload

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class XcpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Preserve upstream BF16/FP32 loading and replace only post-load compute."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def supports_eplb(self) -> bool:
        return False

    def _setup_xcpu_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        import torch_xcpu

        if self.moe.has_bias:
            raise NotImplementedError("XCPU BF16 expert bias is unsupported")
        fused_moe = torch_xcpu.ops.initialize_fused_moe(
            w13,
            w2,
            m_capacity=self.moe.max_num_tokens * self.moe.experts_per_token,
        )
        install_fused_moe(
            self,
            layer,
            fused_moe,
            self.get_fused_moe_quant_config,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if not isinstance(layer, RoutedExperts):
            raise TypeError("XCPU MoE method requires RoutedExperts")
        reject_fused_moe_hot_reload(self)
        # Upstream OOT handling performs common post-load bookkeeping and then
        # deliberately leaves kernel construction to the platform plugin.
        super().process_weights_after_loading(layer)
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
            raise TypeError("XCPU MoE expert weights must be tensors")
        self._setup_xcpu_kernel(layer, w13, w2)
