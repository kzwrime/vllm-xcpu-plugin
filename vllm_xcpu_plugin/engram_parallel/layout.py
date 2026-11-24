"""2D static sharding of Engram tables across all N = DP * TP ranks."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EngramParallelLayout:
    """2D (column-group x row-range) Engram table partition across ``world_size`` ranks.

    All ``L`` Engram layers and ``C`` hash columns per layer form ``M = L * C``
    disjoint prime-sized subtables in a flattened global row space
    ``[0, sum(active_rows))``. With ``G = gcd(M, world_size)`` and
    ``K = world_size // G``, each group of ``M // G`` subtables is cut into
    ``K`` contiguous row ranges across ``K`` ranks.
    """

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    head_sizes: tuple[tuple[int, ...], ...]
    world_size: int
    active_rows: tuple[int, ...]
    layer_base: tuple[int, ...]
    rank_cuts: tuple[int, ...]
    max_cols_per_rank: int
    cols_per_rank: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.layer_ids:
            raise ValueError("EngramParallel needs at least one engram layer")
        if self.world_size <= 0:
            raise ValueError("EngramParallel needs world_size > 0")
        if len(self.num_embeddings) != len(self.layer_ids) or len(
            self.head_sizes
        ) != len(self.layer_ids):
            raise ValueError("Engram layout fields disagree on layer count")
        if len(self.rank_cuts) != self.world_size + 1:
            raise ValueError("rank_cuts length mismatch")
        if len(self.cols_per_rank) != self.world_size:
            raise ValueError("cols_per_rank length mismatch")

    @classmethod
    def from_sizes(
        cls,
        *,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        head_sizes: tuple[tuple[int, ...], ...],
        world_size: int,
    ) -> EngramParallelLayout:
        if not layer_ids:
            raise ValueError("EngramParallel needs at least one engram layer")
        if world_size <= 0:
            raise ValueError("EngramParallel needs world_size > 0")
        if len(num_embeddings) != len(layer_ids) or len(head_sizes) != len(layer_ids):
            raise ValueError("Engram layout fields disagree on layer count")
        total_cols = len(head_sizes[0])
        if total_cols <= 0:
            raise ValueError("EngramParallel needs positive hash column count")

        active_rows: list[int] = []
        layer_base = [0]
        gcol_bounds: list[tuple[int, int]] = []
        for l_idx, (layer_id, sizes) in enumerate(zip(layer_ids, head_sizes)):
            if len(sizes) != total_cols:
                raise ValueError(
                    f"engram layer {layer_id} has {len(sizes)} hash columns, "
                    f"expected {total_cols}"
                )
            offset = 0
            base = layer_base[l_idx]
            for sz in sizes:
                if sz <= 0:
                    raise ValueError(
                        f"engram layer {layer_id} has non-positive bucket size {sz}"
                    )
                gcol_bounds.append((base + offset, base + offset + sz))
                offset += sz
            if offset > num_embeddings[l_idx]:
                raise ValueError(f"engram layer {layer_id} buckets exceed its table")
            active_rows.append(offset)
            layer_base.append(base + offset)

        total_active = layer_base[-1]
        if total_active > (1 << 31) - 1:
            raise ValueError(
                f"total active Engram rows {total_active} exceed int32 range"
            )

        num_subtables = len(layer_ids) * total_cols
        num_groups = math.gcd(num_subtables, world_size)
        k_split = world_size // num_groups
        cols_per_group = num_subtables // num_groups

        rank_cuts = [0] * (world_size + 1)
        for g_idx in range(num_groups):
            col_b = g_idx * cols_per_group
            col_e = (g_idx + 1) * cols_per_group
            group_start = gcol_bounds[col_b][0]
            group_end = gcol_bounds[col_e - 1][1]
            group_rows = group_end - group_start
            for k_idx in range(k_split):
                rank_cuts[g_idx * k_split + k_idx] = (
                    group_start + (k_idx * group_rows) // k_split
                )
        rank_cuts[world_size] = total_active

        cols_per_rank = [0] * world_size
        for g_b, g_e in gcol_bounds:
            for rank in range(world_size):
                if (
                    rank_cuts[rank] < rank_cuts[rank + 1]
                    and rank_cuts[rank + 1] > g_b
                    and rank_cuts[rank] < g_e
                ):
                    cols_per_rank[rank] += 1
        max_cols = max(cols_per_rank)

        return cls(
            layer_ids=tuple(layer_ids),
            num_embeddings=tuple(num_embeddings),
            head_sizes=tuple(tuple(s) for s in head_sizes),
            world_size=world_size,
            active_rows=tuple(active_rows),
            layer_base=tuple(layer_base),
            rank_cuts=tuple(rank_cuts),
            max_cols_per_rank=max_cols,
            cols_per_rank=tuple(cols_per_rank),
        )

    @classmethod
    def from_engram_layout(cls, layout, world_size: int) -> EngramParallelLayout:
        """Build from vLLM's ``EngramLayout`` (``common.engram``)."""
        head_sizes = tuple(
            tuple(size for per_ngram in layer for size in per_ngram)
            for layer in layout.primes
        )
        return cls.from_sizes(
            layer_ids=tuple(layout.layer_ids),
            num_embeddings=tuple(layout.num_embeddings),
            head_sizes=head_sizes,
            world_size=world_size,
        )

    @classmethod
    def from_hf_text_config(
        cls, hf_text_config, world_size: int
    ) -> EngramParallelLayout:
        """Build directly from HF text config without instantiating the model."""
        from vllm.models.deepseek_v41.common.engram import EngramLayout

        layout = EngramLayout.from_config(hf_text_config)
        if layout is None:
            raise ValueError("the model has no engram layers")
        return cls.from_engram_layout(layout, world_size)

    @property
    def layers(self) -> int:
        return len(self.layer_ids)

    @property
    def total_cols(self) -> int:
        return len(self.head_sizes[0])

    @property
    def head_sizes_flat(self) -> tuple[int, ...]:
        return tuple(sz for sizes in self.head_sizes for sz in sizes)

    def layer_slice(self, layer_index: int, rank: int) -> tuple[int, int]:
        """Return ``[lo, hi)`` rows of layer ``layer_index`` owned by ``rank``."""
        if not 0 <= layer_index < self.layers:
            raise ValueError(f"invalid engram layer index {layer_index}")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"invalid EngramParallel rank {rank}")
        my_lo = self.rank_cuts[rank]
        my_hi = self.rank_cuts[rank + 1]
        base = self.layer_base[layer_index]
        lo = max(my_lo, base)
        hi = min(my_hi, self.layer_base[layer_index + 1])
        if lo >= hi:
            return 0, 0
        return lo - base, hi - base

    def layer_slice_by_id(self, layer_id: int, rank: int) -> tuple[int, int, int]:
        """Return ``(lo, hi, expected_full_rows)`` for checkpoint ``layer_id``."""
        layer_index = self.layer_ids.index(layer_id)
        lo, hi = self.layer_slice(layer_index, rank)
        return lo, hi, self.num_embeddings[layer_index]
