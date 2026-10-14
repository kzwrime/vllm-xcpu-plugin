"""Model-worker bootstrap for opt-in AF-EP attention ranks."""

from __future__ import annotations

from typing import Any

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld

from ..common.session_v7 import AfV7Session
from ..common.session_v8 import AfV8Session
from .client_v7 import ExpertsClientV7
from .client_v8 import ExpertsClientV8
from .compatibility import support_from_vllm, validate_attention_support
from .runtime import (
    ExpertsClient,
    get_remote_experts_client,
    register_remote_experts_client,
)


def bootstrap_attention_worker(
    *,
    mpi_world: MpiWorld,
    vllm_config: Any,
    model_world_rank: int,
    model_world_size: int,
) -> ExpertsClient:
    """Validate the worker and assemble its client before model load.

    Version selection belongs here; model
    layers and ModelRunner consume only the runtime's ExpertsClient contract.
    """
    if get_remote_experts_client() is not None:
        raise RuntimeError("an AF-EP attention worker is already registered")
    assert mpi_world.cluster_type == ClusterType.ATTN
    assert mpi_world.cluster_size == model_world_size
    assert mpi_world.cluster_rank == model_world_rank

    backend = vllm_config.parallel_config.all2all_backend
    max_rows = vllm_config.scheduler_config.max_num_batched_tokens
    session: AfV7Session | AfV8Session
    client: ExpertsClient
    if backend == "mpi_alltoallv_v7":
        session = AfV7Session(mpi_world, max_rows_per_attention_rank=max_rows)
        client = ExpertsClientV7(session)
    elif backend == "mpi_alltoallv_v8":
        session = AfV8Session(mpi_world, max_rows_per_attention_rank=max_rows)
        client = ExpertsClientV8(session)
    else:
        raise ValueError(f"unsupported AF-EP backend: {backend}")
    validate_attention_support(
        session.ep_size,
        support_from_vllm(vllm_config),
    )
    register_remote_experts_client(client)
    return client
