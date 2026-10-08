# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_xcpu
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot
class XcpuGateLinear(GateLinear):
    """Retain FP32 router accumulators when the model requests FP32 logits."""

    def forward(self, x):
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
