# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

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
