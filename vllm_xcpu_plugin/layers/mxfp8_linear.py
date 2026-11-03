# SPDX-License-Identifier: Apache-2.0
"""MXFP8 (e4m3 + UE8M0 per 32x32 block) linear layers on XCPU, W8A16.

DeepSeek-V4.1 checkpoints store one UE8M0 scale per 32x32 weight block; the
upstream loader repeats each scale row over its 32 output rows. The weight
stays FP8 and the UE8M0 block scales are passed as stored (the kernels fold
the exponent into the E4M3 -> BF16 conversion), so every projection is a
BF16 x FP8 GEMM with FP32 accumulation:

* dense layers: torch_xcpu FP8 linear with a (32, 32) scale block;
* the grouped wo_a (``xcpu_grouped_wo_a`` = G): ``einsum_fp8_tgd_grd_to_tgr``
  over the [G, R, D] weight, called by the attention layer.

Layers that cannot take that path keep the upstream emulation (BF16 weights
expanded at load).
"""

from typing import Any

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

MXFP8_BLOCK = 32


def _initialize_dummy_ue8m0_scale(scale: torch.Tensor) -> None:
    config = get_current_vllm_config_or_none()
    if config is not None and config.load_config.load_format == "dummy":
        # Dummy loading leaves the exponent bytes uninitialized; 127 is 2^0.
        scale.data.view(torch.uint8).fill_(127)


def _block_scales(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor | None:
    """UE8M0 [N/32, K/32] block scales from the row-repeated scales, or None
    if the layer's scales are not uniform over 32-row blocks."""
    n, k = weight.shape
    if n % MXFP8_BLOCK or k % MXFP8_BLOCK:
        return None
    rows = scale.view(torch.uint8)[:n, : k // MXFP8_BLOCK]
    blocks = rows.view(n // MXFP8_BLOCK, MXFP8_BLOCK, -1)
    if not bool((blocks == blocks[:, :1]).all()):
        return None
    return blocks[:, 0].contiguous().view(torch.float8_e8m0fnu)


class XcpuMxfp8LinearKernel(EmulationMxfp8LinearKernel):
    """BF16 activations against FP8 weights with 32x32 UE8M0 block scales."""

    def process_weights_after_loading(self, layer: Any) -> None:
        import torch_xcpu

        _initialize_dummy_ue8m0_scale(layer.weight_scale)
        weight = layer.weight.data
        scales = None
        if (
            weight.dtype == torch.float8_e4m3fn
            and getattr(layer, "bias", None) is None
            and torch_xcpu.ops.fp8_scale_format_supported(
                (MXFP8_BLOCK, MXFP8_BLOCK), torch.float8_e8m0fnu
            )
        ):
            scales = _block_scales(weight, layer.weight_scale.data)
        groups = getattr(layer, "xcpu_grouped_wo_a", 0)
        if scales is None or (groups and weight.shape[0] % (groups * MXFP8_BLOCK)):
            logger.warning_once(
                "XCPU MXFP8 linear falls back to BF16 emulation for %s",
                tuple(weight.shape),
            )
            super().process_weights_after_loading(layer)
            return

        if groups:
            # Block-diagonal wo_a: [G * R, D] -> [G, R, D] and its block scales.
            n, k = weight.shape
            weight = weight.contiguous().view(groups, n // groups, k)
            packed = torch_xcpu.ops.einsum_fp8_tgd_grd_to_tgr_prepack_weight(weight)
            scales = scales.view(groups, -1, scales.shape[-1])
            layer.xcpu_wo_a = (packed, scales)
            logger.info_once(
                "Using torch_xcpu einsum_fp8_tgd_grd_to_tgr (%s) for wo_a",
                torch_xcpu.ops.einsum_fp8_tgd_grd_to_tgr_version(),
            )
        else:
            linear = torch_xcpu.ops.initialize_fp8_linear(
                weight.contiguous(), scales, (MXFP8_BLOCK, MXFP8_BLOCK)
            )
            packed, scales = linear.params.packed_weight, linear.params.weight_scale
            layer.xcpu_fp8_linear = linear
            logger.info_once(
                "Using torch_xcpu FP8 linear backend=%s with 32x32 UE8M0 blocks",
                linear.params.backend.name.lower(),
            )
        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_scale", scales)

    def apply_weights(
        self,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        linear = getattr(layer, "xcpu_fp8_linear", None)
        if linear is None:
            return super().apply_weights(layer, x, bias)
        assert bias is None, "XCPU MXFP8 linear is initialized without bias"
        return linear(x if x.is_contiguous() else x.contiguous())


def register_mxfp8_linear_kernel() -> None:
    from vllm.model_executor.kernels.linear import register_linear_kernel
    from vllm.platforms import PlatformEnum

    register_linear_kernel(XcpuMxfp8LinearKernel, PlatformEnum.OOT, "mxfp8")
