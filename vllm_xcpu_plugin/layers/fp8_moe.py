"""FP8 checkpoint lifecycle for the unified XCPU MoE pipeline."""

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)

from .fused_moe.grouped_gemm_experts import XcpuGroupedGemmExperts
from .fused_moe.setup import install_fused_moe, reject_fused_moe_hot_reload


class XcpuFp8MoEMethod(Fp8MoEMethod):
    """Load upstream FP8 parameters, then resolve and prepack one XCPU plan."""

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )
        if not quant_config.is_checkpoint_fp8_serialized:
            raise ValueError("XCPU FP8 MoE requires serialized FP8 weights")
        if self.weight_block_size != [128, 128]:
            raise ValueError("XCPU FP8 MoE requires 128x128 block scales")
        if quant_config.activation_scheme != "dynamic":
            raise ValueError("XCPU FP8 MoE requires dynamic activations")
        if layer.moe_config.has_bias:
            raise NotImplementedError("XCPU FP8 MoE expert bias is unsupported")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "XCPU FP8 MoE applies router weights after expert compute"
            )
        supported, reason = XcpuGroupedGemmExperts.is_supported_config(
            XcpuGroupedGemmExperts,
            layer.moe_config,
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            mk.FusedMoEActivationFormat.Standard,
        )
        if not supported:
            raise NotImplementedError(f"XCPU FP8 MoE is unsupported: {reason}")

    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        return fp8_w8a16_moe_quant_config(
            w1_scale=getattr(layer, f"w13_{self.weight_scale_name}"),
            w2_scale=getattr(layer, f"w2_{self.weight_scale_name}"),
            block_shape=self.weight_block_size,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not isinstance(layer, RoutedExperts):
            raise TypeError("XCPU FP8 MoE method requires RoutedExperts")
        reject_fused_moe_hot_reload(self)
        super().process_weights_after_loading(layer)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        import torch_xcpu

        if w13_input_scale is not None or w2_input_scale is not None:
            raise ValueError("XCPU FP8 W8A16 does not use input scales")
        fused_moe = torch_xcpu.ops.initialize_fused_moe(
            w13,
            w2,
            w13_scale,
            w2_scale,
            scale_block_size=self.weight_block_size,
            m_capacity=self.moe.max_num_tokens * self.moe.experts_per_token,
        )
        install_fused_moe(
            self,
            layer,
            fused_moe,
            self.get_fused_moe_quant_config,
            scale_names=(
                f"w13_{self.weight_scale_name}",
                f"w2_{self.weight_scale_name}",
            ),
        )
