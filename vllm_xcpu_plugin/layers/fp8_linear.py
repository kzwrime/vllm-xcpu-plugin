"""vLLM FP8 block-linear kernel backed by torch_xcpu.

The class is intentionally platform-facing glue only. Intel packing and ISA
details stay below torch_xcpu's stable API so the portable simulator and real
NPU implementation can replace the phase-1 backend without changing vLLM.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


class XcpuFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """FP8 W8A16 block-linear kernel for the PrivateUse1 xcpu platform."""

    apply_input_quant = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        import torch_xcpu

        if not torch_xcpu.ops.fp8_scaled_mm_supported():
            return (
                False,
                "No torch_xcpu FP8 block-linear backend is available.",
            )
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        import torch_xcpu

        supported, reason = super().can_implement(config)
        if not supported:
            return supported, reason
        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape.row <= 0 or weight_group_shape.col <= 0:
            return (
                False,
                "torch_xcpu FP8 kernel requires positive scale block sizes, "
                f"got {weight_group_shape}.",
            )
        backend = torch_xcpu.ops.configured_fp8_linear_backend()
        scale_block_n = weight_group_shape.row
        scale_block_k = weight_group_shape.col
        portable_scale_block_supported = (
            scale_block_n in torch_xcpu.ops.PORTABLE_FP8_SCALE_BLOCK_N_INSTANCES
            and scale_block_k == 128
        )
        native_scale_block_supported = (scale_block_n, scale_block_k) == (
            128,
            128,
        )
        if (
            (
                backend
                in (
                    torch_xcpu.ops.Fp8LinearBackend.PORTABLE_LUT,
                    torch_xcpu.ops.Fp8LinearBackend.PORTABLE_DIRECT,
                    torch_xcpu.ops.Fp8LinearBackend.PORTABLE_PRE_SCALED,
                )
                and not portable_scale_block_supported
            )
            or (
                backend
                not in (
                    torch_xcpu.ops.Fp8LinearBackend.PORTABLE_LUT,
                    torch_xcpu.ops.Fp8LinearBackend.PORTABLE_DIRECT,
                    torch_xcpu.ops.Fp8LinearBackend.PORTABLE_PRE_SCALED,
                )
                and not native_scale_block_supported
            )
            or config.weight_shape[0] % 32 != 0
            or config.weight_shape[1] % 128 != 0
        ):
            return (
                False,
                "torch_xcpu portable FP8 backends instantiate scale_block_n "
                "in {32, 64, 128, 256}; native backends require a 128x128 "
                "scale block. All backends require scale_block_k=128, "
                "N % 32 == 0 and K % 128 == 0; got scale block "
                f"{weight_group_shape} and weight shape {config.weight_shape}.",
            )
        if config.out_dtype != torch.bfloat16:
            return False, "torch_xcpu FP8 kernel requires BF16 output."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import torch_xcpu

        params = self._get_layer_params(layer)
        scale_attr = (
            params.WEIGHT_SCALE_INV
            if params.weight_scale_inv is not None
            else params.WEIGHT_SCALE
        )
        weight_scale = (
            params.weight_scale_inv
            if params.weight_scale_inv is not None
            else params.weight_scale
        )
        assert weight_scale is not None
        weight_scale = weight_scale.data.contiguous()

        layer_bias = getattr(layer, "bias", None)
        bias_fp32 = None
        if layer_bias is not None:
            assert isinstance(layer_bias, torch.Tensor)
            bias_fp32 = layer_bias.float().contiguous()

        fp8_linear = torch_xcpu.ops.initialize_fp8_linear(
            params.weight,
            weight_scale,
            tuple(self.weight_group_shape),
            bias_fp32,
        )
        logger.warning_once(
            "Using torch_xcpu FP8 linear backend=%s",
            fp8_linear.params.backend.name.lower(),
            scope="process",
        )

        replace_parameter(
            layer,
            params.WEIGHT,
            fp8_linear.params.packed_weight,
        )
        replace_parameter(
            layer,
            scale_attr,
            fp8_linear.params.weight_scale,
        )
        if bias_fp32 is not None:
            replace_parameter(layer, "bias", bias_fp32)

        # Keep the initialized implementation on the layer, rather than on the
        # kernel instance, so its packed parameters cannot be mixed up if vLLM
        # ever shares a kernel object between multiple linear layers.
        layer._xcpu_fp8_linear = fp8_linear

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        fp8_linear = getattr(layer, "_xcpu_fp8_linear", None)
        if fp8_linear is None:
            raise RuntimeError(
                "torch_xcpu FP8 linear was not initialized; "
                "process_weights_after_loading() must run before inference"
            )
        if (bias is not None) != (fp8_linear.params.bias is not None):
            raise RuntimeError(
                "The runtime bias does not match the bias used to initialize "
                "torch_xcpu FP8 linear"
            )
        return fp8_linear(x)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "XcpuFp8BlockScaledMMLinearKernel overrides apply_weights directly."
        )


def register_fp8_linear_kernel() -> None:
    """Register the block-FP8 kernel through vLLM's public OOT API."""
    from vllm.model_executor.kernels.linear import register_linear_kernel
    from vllm.platforms.interface import PlatformEnum

    register_linear_kernel(
        XcpuFp8BlockScaledMMLinearKernel,
        PlatformEnum.OOT,
        "fp8_block",
    )
