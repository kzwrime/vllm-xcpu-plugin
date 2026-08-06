"""vLLM MXFP4 W4A16 linear kernel backed by ``torch_xcpu``."""

import torch
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp4 import (
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


class XcpuMxFp4LinearKernel(MxFp4LinearKernel):
    """Keep checkpoint interpretation in vLLM and execution in torch_xcpu."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        import torch_xcpu

        if not torch_xcpu.ops.mxfp4_scaled_mm_supported():
            return False, "No torch_xcpu MXFP4 linear backend is available."
        return True, None

    @classmethod
    def can_implement(
        cls, config: MxFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import torch_xcpu

        bias = getattr(layer, "bias", None)
        bias_fp32 = None if bias is None else bias.float().contiguous()
        linear = torch_xcpu.ops.initialize_mxfp4_linear(
            layer.weight,
            layer.weight_scale,
            bias_fp32,
        )
        logger.warning_once(
            "Using torch_xcpu MXFP4 linear backend=%s",
            linear.params.backend.name.lower(),
            scope="process",
        )
        replace_parameter(layer, "weight", linear.params.packed_weight)
        replace_parameter(
            layer,
            "weight_scale",
            linear.params.packed_weight_scale,
        )
        if bias_fp32 is not None:
            replace_parameter(layer, "bias", bias_fp32)
        layer._xcpu_mxfp4_linear = linear

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        linear = getattr(layer, "_xcpu_mxfp4_linear", None)
        if linear is None:
            raise RuntimeError(
                "torch_xcpu MXFP4 linear was not initialized; "
                "process_weights_after_loading() must run before inference"
            )
        if (bias is not None) != (linear.params.bias is not None):
            raise RuntimeError(
                "The runtime bias does not match the bias used to initialize "
                "torch_xcpu MXFP4 linear"
            )
        return linear(x)


def register_mxfp4_linear_kernel() -> None:
    """Register MXFP4 through vLLM's public OOT linear-kernel API."""
    from vllm.model_executor.kernels.linear import register_linear_kernel
    from vllm.platforms.interface import PlatformEnum

    register_linear_kernel(
        XcpuMxFp4LinearKernel,
        PlatformEnum.OOT,
        "mxfp4",
    )
