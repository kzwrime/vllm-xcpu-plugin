"""Platform-scoped RoutedExperts replacement for XCPU MoE methods."""

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.platforms import current_platform


def xcpu_moe_method_factory(
    upstream: FusedMoEMethodBase,
    quant_config: QuantizationConfig | None,
    layer: RoutedExperts,
) -> FusedMoEMethodBase:
    """Map only the three supported upstream MoE method families."""
    if isinstance(upstream, Fp8MoEMethod):
        if not isinstance(quant_config, Fp8Config):
            raise TypeError("upstream FP8 MoE method requires Fp8Config")
        from vllm_xcpu_plugin.layers.fp8_moe import XcpuFp8MoEMethod

        return XcpuFp8MoEMethod(quant_config, layer)
    if isinstance(upstream, CompressedTensorsW4A4Mxfp4MoEMethod):
        from vllm_xcpu_plugin.layers.mxfp4_moe import (
            XcpuCompressedTensorsMxfp4MoEMethod,
        )

        return XcpuCompressedTensorsMxfp4MoEMethod(layer.moe_config)
    if isinstance(upstream, QuarkOCP_MX_MoEMethod):
        from vllm_xcpu_plugin.layers.quark_mxfp4 import (
            XcpuQuarkOCPMXMoEMethod,
            should_use_quark_mxfp4_w4a16,
        )

        if not should_use_quark_mxfp4_w4a16(
            upstream.weight_dtype, upstream.input_dtype
        ):
            return upstream

        return XcpuQuarkOCPMXMoEMethod(
            upstream.weight_quant,
            None,
            layer.moe_config,
        )
    if isinstance(upstream, UnquantizedFusedMoEMethod):
        from .unquantized_fused_moe_method import (
            XcpuUnquantizedFusedMoEMethod,
        )

        return XcpuUnquantizedFusedMoEMethod(layer.moe_config)
    return upstream


@RoutedExperts.register_oot
class XcpuRoutedExperts(RoutedExperts):
    """Delegate checkpoint interpretation upstream, then replace MoE compute."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        upstream = super()._get_quant_method(prefix, quant_config, moe_config)
        if current_platform.device_name != "mcpu":
            return upstream
        return xcpu_moe_method_factory(upstream, quant_config, self)
