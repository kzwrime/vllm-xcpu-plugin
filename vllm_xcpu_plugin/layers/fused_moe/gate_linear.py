# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot
class XcpuGateLinear(GateLinear):
    """Retain FP32 router accumulators when the model requests FP32 logits."""

    def forward(self, x):
        import torch_xcpu

        # VLLM_USE_XCPU_LINEAR prepacked this gate into the torch_xcpu BF16
        # linear backend (BF16@BF16 -> BF16/FP32 per self.out_dtype) and freed
        # the raw weight; the packed path already keeps FP32 accumulators.
        bf16_linear = getattr(self, "_xcpu_bf16_linear", None)
        if bf16_linear is not None:
            return bf16_linear(x), None
        if (
            self.out_dtype == torch.float32
            and self.bias is None
            and x.ndim == 2
            and x.device.type in ("mcpu", "privateuseone")
            and self.weight.device == x.device
            and x.dtype == self.weight.dtype
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            output = torch_xcpu.ops.mm(x, self.weight.T, out_dtype=torch.float32)
            return output, None
        return super().forward(x)
