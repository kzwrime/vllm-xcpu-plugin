"""Engram disaggregation state shared by the attention and Engram roles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_xcpu_plugin.disagg import PairSession
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld


@dataclass(frozen=True)
class EngramShardLayout:
    """How the engram tables are split over the Engram ranks.

    Every engram layer's hash columns (``(max_ngram - 1) * n_heads``, in the
    ``EngramLayout.primes`` order) are cut into ``num_engram_ranks`` equal
    contiguous groups; rank ``e`` serves columns ``[e * cols_per_rank,
    (e + 1) * cols_per_rank)`` of every layer, i.e. the rows
    ``[vocab_start, vocab_end)`` of each layer's table.
    """

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    # Per layer, the bucket row count of every hash column (flattened primes).
    head_sizes: tuple[tuple[int, ...], ...]
    num_engram_ranks: int

    def __post_init__(self) -> None:
        if not self.layer_ids:
            raise ValueError("Engram disaggregation needs at least one engram layer")
        if len(self.num_embeddings) != len(self.layer_ids) or len(
            self.head_sizes
        ) != len(self.layer_ids):
            raise ValueError("Engram layout fields disagree on the layer count")
        if self.num_engram_ranks <= 0:
            raise ValueError("Engram disaggregation needs at least one Engram rank")
        if self.total_cols % self.num_engram_ranks:
            raise ValueError(
                f"{self.total_cols} engram hash columns cannot be split evenly over "
                f"{self.num_engram_ranks} Engram ranks"
            )
        for layer, sizes in zip(self.layer_ids, self.head_sizes):
            if len(sizes) != self.total_cols:
                raise ValueError(f"engram layer {layer} has {len(sizes)} hash columns")
            if sum(sizes) > self.num_embeddings[self.layer_ids.index(layer)]:
                raise ValueError(f"engram layer {layer} buckets exceed its table")

    @classmethod
    def from_engram_layout(cls, layout, num_engram_ranks: int) -> EngramShardLayout:
        """From vLLM's ``EngramLayout`` (``common.engram``)."""
        head_sizes = tuple(
            tuple(size for per_ngram in layer for size in per_ngram)
            for layer in layout.primes
        )
        return cls(
            layer_ids=tuple(layout.layer_ids),
            num_embeddings=tuple(layout.num_embeddings),
            head_sizes=head_sizes,
            num_engram_ranks=num_engram_ranks,
        )

    @property
    def layers(self) -> int:
        return len(self.layer_ids)

    @property
    def total_cols(self) -> int:
        return len(self.head_sizes[0])

    @property
    def cols_per_rank(self) -> int:
        return self.total_cols // self.num_engram_ranks

    def column_range(self, layer_index: int, engram_rank: int) -> tuple[int, int]:
        """Rows ``[vocab_start, vocab_end)`` of layer ``layer_index`` that
        Engram rank ``engram_rank`` serves."""
        if not 0 <= engram_rank < self.num_engram_ranks:
            raise ValueError(f"invalid Engram rank {engram_rank}")
        sizes = self.head_sizes[layer_index]
        first = engram_rank * self.cols_per_rank
        start = sum(sizes[:first])
        return start, start + sum(sizes[first : first + self.cols_per_rank])


class EngramSession(PairSession):
    """One rank's view of the attention/Engram pair and its transport metadata.

    Consumes an already connected MPI world; ``initialize()`` runs the
    collective layout check and capacity negotiation through torch_xcpu.
    """

    peer_role = ClusterType.ENGRAM
    name = "Engram disaggregation"

    def __init__(self, mpi_world: MpiWorld, *, max_tokens_proposal: int) -> None:
        from torch_xcpu import ops as xcpu_ops

        self.protocol_version = xcpu_ops.ENGRAM_REMOTE_PROTOCOL
        super().__init__(mpi_world)
        if self.is_attention:
            if max_tokens_proposal <= 0:
                raise ValueError("attention ranks must propose a positive capacity")
        elif max_tokens_proposal != 0:
            raise ValueError("Engram ranks must not propose a capacity")
        self.num_engram_ranks = self.num_peer_ranks
        self.max_tokens_proposal = max_tokens_proposal
        self.shard: EngramShardLayout | None = None
        self.max_tokens: int | None = None

    def initialize(self, shard: EngramShardLayout) -> int:
        """Collective over the attention/Engram pair; returns the negotiated
        capacity."""
        if self.shard is not None:
            if self.shard != shard:
                raise RuntimeError(
                    "the Engram shard layout changed after initialization"
                )
            assert self.max_tokens is not None
            return self.max_tokens
        if shard.num_engram_ranks != self.num_engram_ranks:
            raise ValueError(
                f"shard layout expects {shard.num_engram_ranks} Engram ranks, the "
                f"MPI world has {self.num_engram_ranks}"
            )
        from torch_xcpu import ops as xcpu_ops

        if self.is_attention:
            # Loading may have queued device work; finish it before MPI.
            torch.accelerator.synchronize()
        self.max_tokens = xcpu_ops.deepseek_v41_engram_remote_initialize(
            self.metadata,
            self.communicator_handle,
            self.max_tokens_proposal,
            shard.layers,
            shard.total_cols,
            shard.cols_per_rank,
            list(shard.num_embeddings),
        )
        self.shard = shard
        print(
            f"Engram disaggregation ready: role={self.cluster_type.name} "
            f"rank={self.role_rank} attention_ranks={self.num_attention_ranks} "
            f"engram_ranks={self.num_engram_ranks} layers={shard.layer_ids} "
            f"cols_per_rank={shard.cols_per_rank} max_tokens={self.max_tokens}",
            flush=True,
        )
        return self.max_tokens

    def require_initialized(self) -> tuple[EngramShardLayout, int]:
        if self.shard is None or self.max_tokens is None:
            raise RuntimeError("the Engram session has not been initialized")
        return self.shard, self.max_tokens

    def finalize(self) -> None:
        """Forget the negotiated layout after the native endpoint context is
        gone (``deepseek_v41_engram_remote_cleanup``): a later ``initialize``
        must renegotiate through the native collective instead of answering
        from a cache that no longer has a backend."""
        self.shard = None
        self.max_tokens = None
