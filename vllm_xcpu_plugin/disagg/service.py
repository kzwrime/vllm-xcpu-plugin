"""Peer-rank process skeleton: join the world, load, serve, exit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_xcpu_plugin.distributed.mpi_world import (
    ClusterType,
    MpiWorld,
    initialize_mpi,
    initialize_mpi_world,
)


def join_mpi_world(cluster_type: ClusterType) -> tuple[Any, MpiWorld]:
    """Enter the MPMD world as ``cluster_type``.

    The attention ranks' ``run_mp_rpc_worker`` executes a world barrier before
    importing vLLM; every peer rank must keep that collective order.
    """
    mpi = initialize_mpi()
    mpi.COMM_WORLD.Barrier()
    return mpi, initialize_mpi_world(cluster_type)


def run_role_service(
    cluster_type: ClusterType,
    load: Callable[[MpiWorld], Any],
    serve: Callable[[Any], None],
    *,
    finalize: bool,
) -> None:
    """Load this rank's state (aborting the world on failure, since the
    attention ranks may already wait in initialization), serve until the
    service returns, then optionally finalize MPI with a world barrier.
    """
    mpi, mpi_world = join_mpi_world(cluster_type)
    try:
        state = load(mpi_world)
    except Exception:
        import traceback

        traceback.print_exc()
        mpi.COMM_WORLD.Abort(1)
        raise
    serve(state)
    if finalize:
        mpi_world.global_world_comm.Barrier()
        mpi.Finalize()
