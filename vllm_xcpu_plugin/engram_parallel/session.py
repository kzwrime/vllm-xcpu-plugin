"""MPI communicator lifecycle and layout handshake for EngramParallel."""

from __future__ import annotations

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld

from .layout import EngramParallelLayout


class EngramParallelSession:
    """One model rank's view of the duplicated EP/world EngramParallel communicator."""

    def __init__(
        self,
        mpi_world: MpiWorld,
        *,
        world_rank: int,
        world_size: int,
        max_tokens_proposal: int,
    ) -> None:
        if mpi_world.cluster_type != ClusterType.ATTN:
            raise ValueError("EngramParallel runs on ATTN worker ranks only")
        if (
            mpi_world.cluster_size != world_size
            or mpi_world.cluster_rank != world_rank
        ):
            raise ValueError(
                f"EngramParallel rank mismatch: mpi cluster={mpi_world.cluster_rank}/"
                f"{mpi_world.cluster_size}, worker={world_rank}/{world_size}"
            )
        if max_tokens_proposal <= 0:
            raise ValueError("EngramParallel requires positive max_tokens_proposal")

        self.world_rank = world_rank
        self.world_size = world_size
        self.max_tokens_proposal = max_tokens_proposal
        self._comm = mpi_world.cluster_comm.Dup()
        self.communicator_handle = torch.tensor(
            [self._comm.py2f()], dtype=torch.int64, device="cpu"
        )
        self.layout: EngramParallelLayout | None = None
        self.max_tokens: int | None = None

    def initialize(
        self,
        layout: EngramParallelLayout,
        weights: list[torch.Tensor],
        scales: list[torch.Tensor],
        *,
        max_tokens_proposal: int | None = None,
        dp_size: int = 0,
        dp_rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        use_sequence_parallel: bool = False,
    ) -> int:
        """Collective over the duplicated communicator; returns negotiated capacity."""
        if self._comm is None:
            raise RuntimeError("EngramParallel session has already been closed")
        if layout.world_size != self.world_size:
            raise ValueError(
                f"layout world_size={layout.world_size} != session {self.world_size}"
            )
        if self.layout is not None:
            if self.layout != layout:
                raise RuntimeError(
                    "EngramParallel layout changed after initialization"
                )
            assert self.max_tokens is not None
            return self.max_tokens

        proposal = (
            self.max_tokens_proposal
            if max_tokens_proposal is None
            else max_tokens_proposal
        )
        if proposal <= 0:
            raise ValueError("EngramParallel max_tokens proposal must be positive")

        from torch_xcpu import ops as xcpu_ops

        torch.accelerator.synchronize()
        self.max_tokens = xcpu_ops.deepseek_v41_engram_parallel_initialize(
            self.communicator_handle,
            self.world_size,
            self.world_rank,
            proposal,
            layout.layers,
            layout.total_cols,
            list(layout.head_sizes_flat),
            list(layout.num_embeddings),
            weights,
            scales,
            dp_size=dp_size,
            dp_rank=dp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_sequence_parallel=use_sequence_parallel,
        )
        self.layout = layout
        print(
            f"EngramParallel ready: rank={self.world_rank}/{self.world_size} "
            f"layers={layout.layer_ids} max_cols_per_rank={layout.max_cols_per_rank} "
            f"max_tokens={self.max_tokens}",
            flush=True,
        )
        return self.max_tokens

    def require_initialized(self) -> tuple[EngramParallelLayout, int]:
        if self.layout is None or self.max_tokens is None:
            raise RuntimeError("EngramParallel session has not been initialized")
        return self.layout, self.max_tokens

    def finalize(self) -> None:
        self.layout = None
        self.max_tokens = None

    def _teardown_comm(self, action: str, *args: object) -> None:
        self.finalize()
        if self._comm is not None:
            from mpi4py import MPI

            if (
                not isinstance(self._comm, MPI.Comm)
                or (MPI.Is_initialized() and not MPI.Is_finalized())
            ):
                getattr(self._comm, action)(*args)
            self._comm = None

    def close(self) -> None:
        self._teardown_comm("Free")

    def abort(self, errorcode: int = 1) -> None:
        self._teardown_comm("Abort", errorcode)
