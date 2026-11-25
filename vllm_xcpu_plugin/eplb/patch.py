# SPDX-License-Identifier: Apache-2.0
"""Install XCPU EPLB router op rebind and MPI communicator factory."""

from __future__ import annotations

import torch
import vllm.model_executor.layers.fused_moe.router.base_router as base_router
from vllm.distributed.eplb.eplb_communicator import (
    register_eplb_communicator_factory,
)

from vllm_xcpu_plugin.distributed.mpi_eplb_communicator import MpiEplbCommunicator
from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility
from vllm_xcpu_plugin.upstream_contracts import rebind


def _xcpu_eplb_map_to_physical_and_record(
    topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    import torch_xcpu

    if topk_ids.device.type not in ("cpu", "privateuseone", "mcpu"):
        raise NotImplementedError(
            "XCPU EPLB mapping supports only CPU or mcpu tensors, "
            f"got {topk_ids.device}"
        )
    return torch_xcpu.ops.eplb_map_to_physical_and_record(
        topk_ids=topk_ids,
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
        record_enabled=record_enabled,
        num_unpadded_tokens=num_unpadded_tokens,
    )


def install_eplb() -> None:
    """Verify EPLB contracts, register MPI comm, and rebind router op."""
    verify_upstream_compatibility(("eplb",))
    register_eplb_communicator_factory("mpi", MpiEplbCommunicator)
    rebind(
        base_router,
        "eplb_map_to_physical_and_record",
        _xcpu_eplb_map_to_physical_and_record,
    )
