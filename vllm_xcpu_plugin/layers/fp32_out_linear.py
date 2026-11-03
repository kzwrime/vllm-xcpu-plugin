# SPDX-License-Identifier: Apache-2.0
"""Unquantized BF16 linears with FP32 output on XCPU.

The DeepSeek-V4.1 compressor kv/score projection keeps the FP32 accumulator
(``torch.mm(x, W.T, out_dtype=fp32)`` upstream). The torch_xcpu BF16 linear
returns that accumulator directly through ``layer._xcpu_bf16_linear``.
"""

from typing import Any

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.utils import replace_parameter


class XcpuFp32OutLinearMethod(UnquantizedLinearMethod):
    """UnquantizedLinearMethod on the torch_xcpu BF16 linear, FP32 output."""

    def process_weights_after_loading(self, layer: Any) -> None:
        import torch_xcpu

        assert getattr(layer, "bias", None) is None, "FP32-out linear takes no bias"
        layer.out_dtype = torch.float32
        super().process_weights_after_loading(layer)
        if getattr(layer, "_xcpu_bf16_linear", None) is not None:
            return
        if (
            layer.weight.ndim == 2
            and layer.weight.dtype == torch.bfloat16
            and layer.weight.device.type in ("privateuseone", "mcpu")
            and torch_xcpu.ops.bf16_linear_supported()
            and torch_xcpu.ops.bf16_linear_shape_supported(
                layer.weight.size(0), layer.weight.size(1)
            )
        ):
            linear = torch_xcpu.ops.initialize_bf16_linear(
                layer.weight.data, out_dtype=torch.float32
            )
            replace_parameter(layer, "weight", linear.params.packed_weight)
            layer._xcpu_bf16_linear = linear

    def apply(
        self,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_xcpu

        assert bias is None
        bf16_linear = getattr(layer, "_xcpu_bf16_linear", None)
        if bf16_linear is not None:
            return bf16_linear(x)
        return torch_xcpu.ops.mm(x, layer.weight.T, out_dtype=torch.float32)
