"""Quark MXFP4 W4A16 bridges backed by ``torch_xcpu``."""

from typing import Any, cast

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.utils import replace_parameter

from .fused_moe.grouped_gemm_experts import XcpuGroupedGemmExperts
from .fused_moe.setup import install_fused_moe, reject_fused_moe_hot_reload

logger = init_logger(__name__)


def is_quark_mxfp4_w4a16(
    weight_dtype: str,
    input_dtype: str | None,
) -> bool:
    """Return whether the checkpoint natively declares MXFP4 weight-only."""
    return weight_dtype == "mxfp4" and input_dtype is None


def _initialize_dummy_e8m0_scale(scale: torch.Tensor) -> None:
    config = get_current_vllm_config_or_none()
    if config is not None and config.load_config.load_format == "dummy":
        # DummyModelLoader intentionally leaves integer tensors uninitialized.
        # E8M0 bias 127 represents a finite unit scale; 255 is reserved.
        scale.fill_(127)


class XcpuQuarkOCPMXLinearScheme(QuarkOCP_MX):
    """Execute Quark MXFP4 weights with BF16 activations in torch_xcpu."""

    def __init__(
        self,
        weight_quant_spec,
        input_quant_spec,
        dynamic_mxfp4_quant: bool = False,
    ):
        super().__init__(
            weight_quant_spec,
            input_quant_spec,
            dynamic_mxfp4_quant=dynamic_mxfp4_quant,
        )
        self._use_xcpu_w4a16 = (
            is_quark_mxfp4_w4a16(self.weight_dtype, self.input_dtype)
            and not self.dynamic_mxfp4_quant
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self._use_xcpu_w4a16:
            return super().process_weights_after_loading(layer)

        import torch_xcpu

        weight_scale = layer.weight_scale
        assert isinstance(weight_scale, torch.Tensor)
        _initialize_dummy_e8m0_scale(weight_scale)
        bias = getattr(layer, "bias", None)
        bias_fp32 = None if bias is None else bias.float().contiguous()
        linear = torch_xcpu.ops.initialize_mxfp4_linear(
            layer.weight,
            weight_scale,
            bias_fp32,
        )
        logger.warning_once(
            "Using torch_xcpu for Quark MXFP4 W4A16 linear: backend=%s",
            linear.params.backend.name.lower(),
            scope="process",
        )
        replace_parameter(layer, "weight", linear.params.packed_weight)
        replace_parameter(layer, "weight_scale", linear.params.packed_weight_scale)
        if bias_fp32 is not None:
            replace_parameter(layer, "bias", bias_fp32)
        layer._xcpu_quark_mxfp4_linear = linear

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._use_xcpu_w4a16:
            return super().apply_weights(layer, x, bias)
        linear = getattr(layer, "_xcpu_quark_mxfp4_linear", None)
        if linear is None:
            raise RuntimeError(
                "torch_xcpu Quark MXFP4 linear was not initialized; "
                "process_weights_after_loading() must run before inference"
            )
        if (bias is not None) != (linear.params.bias is not None):
            raise RuntimeError(
                "The runtime bias does not match the bias used to initialize "
                "torch_xcpu Quark MXFP4 linear"
            )
        return linear(x)


class XcpuQuarkOCPMXMoEMethod(QuarkOCP_MX_MoEMethod):
    """Keep Quark's W4A16 parameter schema and use XCPU grouped GEMM."""

    def __init__(self, weight_config, input_config, moe):
        super().__init__(weight_config, input_config, moe)
        if self.weight_dtype != "mxfp4" or self.input_dtype is not None:
            raise ValueError("XCPU Quark MXFP4 MoE bridge only supports W4A16")
        self.mxfp4_backend = Mxfp4MoeBackend.CPU
        self.experts_cls = XcpuGroupedGemmExperts

    @property
    def supports_eplb(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import torch_xcpu

        if not isinstance(layer, RoutedExperts):
            raise TypeError("XCPU Quark MXFP4 MoE method requires RoutedExperts")
        reject_fused_moe_hot_reload(self)
        if self.has_bias:
            raise NotImplementedError("XCPU Quark MXFP4 expert bias is unsupported")
        w13_weight_scale = layer.w13_weight_scale
        w2_weight_scale = layer.w2_weight_scale
        assert isinstance(w13_weight_scale, torch.Tensor)
        assert isinstance(w2_weight_scale, torch.Tensor)
        _initialize_dummy_e8m0_scale(w13_weight_scale)
        _initialize_dummy_e8m0_scale(w2_weight_scale)
        fused_moe = torch_xcpu.ops.initialize_fused_moe(
            layer.w13_weight,
            layer.w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            scale_block_size=(1, 32),
            m_capacity=self.moe.max_num_tokens * self.moe.experts_per_token,
        )
        install_fused_moe(
            self,
            layer,
            fused_moe,
            lambda current_layer: make_mxfp4_moe_quant_config(
                mxfp4_backend=Mxfp4MoeBackend.CPU,
                w1_scale=current_layer.w13_weight_scale,
                w2_scale=current_layer.w2_weight_scale,
                gemm1_alpha=getattr(current_layer, "swiglu_alpha", None),
                gemm1_beta=getattr(current_layer, "swiglu_beta", None),
                swiglu_limit=getattr(current_layer, "swiglu_limit", None),
                layer=current_layer,
            ),
            scale_names=("w13_weight_scale", "w2_weight_scale"),
        )


def register_quark_mxfp4_linear_scheme() -> None:
    """Install the only missing Quark linear-scheme extension point."""
    import vllm.model_executor.layers.quantization.quark.quark as quark

    cast(Any, quark).QuarkOCP_MX = XcpuQuarkOCPMXLinearScheme
