"""MXFP4 checkpoint lifecycle for the unified XCPU MoE pipeline."""

import torch
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)

from .fused_moe.grouped_gemm_experts import XcpuGroupedGemmExperts
from .fused_moe.setup import install_fused_moe, reject_fused_moe_hot_reload


class XcpuCompressedTensorsMxfp4MoEMethod(CompressedTensorsW4A4Mxfp4MoEMethod):
    """Keep upstream MXFP4 parameter schema and replace its compute backend."""

    def __init__(self, moe):
        super().__init__(moe)
        self.use_cutlass_mxfp4 = False
        self.mxfp4_backend = Mxfp4MoeBackend.CPU
        self.experts_cls = XcpuGroupedGemmExperts

    @property
    def supports_eplb(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import torch_xcpu

        if not isinstance(layer, RoutedExperts):
            raise TypeError("XCPU MXFP4 MoE method requires RoutedExperts")
        reject_fused_moe_hot_reload(self)
        if self.moe.has_bias:
            raise NotImplementedError("XCPU MXFP4 expert bias is unsupported")
        fused_moe = torch_xcpu.ops.initialize_fused_moe(
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            scale_block_size=(1, 32),
            m_capacity=self.moe.max_num_tokens * self.moe.experts_per_token,
        )
        del layer.w13_weight_packed
        del layer.w2_weight_packed
        install_fused_moe(
            self,
            layer,
            fused_moe,
            lambda current_layer: make_mxfp4_moe_quant_config(
                mxfp4_backend=Mxfp4MoeBackend.CPU,
                w1_scale=current_layer.w13_weight_scale,
                w2_scale=current_layer.w2_weight_scale,
                layer=current_layer,
            ),
            scale_names=("w13_weight_scale", "w2_weight_scale"),
        )
