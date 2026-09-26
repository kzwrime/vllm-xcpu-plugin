# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_shared,
)


@MoERunner.register_oot
class XcpuMoERunner(MoERunner):
    """Expose the MoE implementation to Dynamo on XCPU.

    The upstream runner normally enters an opaque ``torch.ops.vllm.moe_forward``
    custom op. That custom op executes its Python implementation after the
    model graph has been compiled, so router/dispatch/experts/finalize remain
    outside the AOT graph. XCPU uses the direct entry while Dynamo is tracing;
    eager execution remains semantically identical.
    """

    def _select_forward(self) -> Callable:
        return _moe_forward if self._shared_experts is None else _moe_forward_shared

    def _combine_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        if (
            shared_output is not None
            and self.routed_scaling_factor != 1.0
            and self.routed_output_transform is None
            and fused_output.device.type == "mcpu"
            and shared_output.device == fused_output.device
            and fused_output.dtype == shared_output.dtype == torch.bfloat16
            and fused_output.ndim == 2
            and shared_output.shape == fused_output.shape
            and fused_output.is_contiguous()
            and shared_output.is_contiguous()
        ):
            from torch_xcpu.ops import moe_scaled_add

            return moe_scaled_add(
                shared_output, fused_output, self.routed_scaling_factor
            )
        return super()._combine_shared_expert_output(shared_output, fused_output)
