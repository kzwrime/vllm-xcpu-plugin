"""vLLM FP8 block-linear kernel backed by torch_xcpu.

The class is intentionally platform-facing glue only. Intel packing and ISA
details stay below torch_xcpu's stable API so the portable simulator and real
NPU implementation can replace the phase-1 backend without changing vLLM.
"""

import torch
from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.utils import replace_parameter


class XcpuFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """FP8 W8A16 block-linear kernel for the PrivateUse1 xcpu platform."""

    apply_input_quant = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        import torch_xcpu

        if not torch_xcpu.ops.fp8_block_linear_supported():
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
        backend = torch_xcpu.ops.resolve_fp8_linear_backend()
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
            (backend.startswith("portable_") and not portable_scale_block_supported)
            or (
                not backend.startswith("portable_") and not native_scale_block_supported
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
            layer.register_parameter(
                "bias_fp32",
                torch.nn.Parameter(bias_fp32, requires_grad=False),
            )

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
    """Register through vLLM's OOT plugin lifecycle.

    vLLM exposes ``register_linear_kernel`` publicly, but the version bundled
    with this kit does not yet accept its own block-FP8 registry as a kernel
    type. Prefer that public API once it does; keep the version-specific
    fallback contained here so no vLLM source patch is required.
    """
    from vllm.model_executor.kernels import linear
    from vllm.platforms.interface import PlatformEnum

    try:
        linear.register_linear_kernel(
            XcpuFp8BlockScaledMMLinearKernel,
            PlatformEnum.OOT,
            "fp8_block",
        )
        return
    except ValueError as exc:
        if "Unrecognized kernel type" not in str(exc):
            raise

    # Compatibility fallback for vLLM
    # 4b2dd5f509a2ee3d5ec1c0c9832a89a0cb19072d. This is registry extension,
    # not monkey-patching behavior or modifying the vLLM checkout.
    kernels = linear._POSSIBLE_FP8_BLOCK_KERNELS.setdefault(PlatformEnum.OOT, [])
    if XcpuFp8BlockScaledMMLinearKernel not in kernels:
        kernels.insert(0, XcpuFp8BlockScaledMMLinearKernel)
