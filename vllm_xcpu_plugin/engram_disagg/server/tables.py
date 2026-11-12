"""Load one Engram rank's table columns from the checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from ..common import EngramShardLayout

# vLLM's dummy loader fills the tables with these (``dummy_weight_value``
# on ParallelEngramEmbedding): E4M3 1.0 and the UE8M0 byte of scale 1.0.
DUMMY_WEIGHT_BYTE = (
    torch.tensor(1.0, dtype=torch.float8_e4m3fn).view(torch.uint8).item()
)
DUMMY_SCALE_BYTE = 127
SCALE_BYTES_PER_ROW = 256 // 32


@dataclass(frozen=True)
class EngramShardTables:
    weights: list[torch.Tensor]  # per layer, uint8 [rows, 256]
    scales: list[torch.Tensor]  # per layer, uint8 [rows, 8]
    vocab_starts: list[int]


def table_names(layer_id: int) -> tuple[str, str]:
    return (
        f"layers.{layer_id}.engram.embed.weight",
        f"layers.{layer_id}.engram.embed.scale",
    )


def load_engram_shard(
    shard: EngramShardLayout,
    engram_rank: int,
    *,
    model: str,
    revision: str | None,
    load_format: str,
) -> EngramShardTables:
    ranges = [shard.column_range(i, engram_rank) for i in range(shard.layers)]
    if load_format == "dummy":
        return _dummy_tables(ranges)
    return _checkpoint_tables(shard, ranges, model=model, revision=revision)


def _dummy_tables(ranges: list[tuple[int, int]]) -> EngramShardTables:
    weights, scales = [], []
    for start, end in ranges:
        weights.append(
            torch.full((end - start, 256), DUMMY_WEIGHT_BYTE, dtype=torch.uint8)
        )
        scales.append(
            torch.full(
                (end - start, SCALE_BYTES_PER_ROW), DUMMY_SCALE_BYTE, dtype=torch.uint8
            )
        )
    return EngramShardTables(weights, scales, [start for start, _ in ranges])


def _checkpoint_tables(
    shard: EngramShardLayout,
    ranges: list[tuple[int, int]],
    *,
    model: str,
    revision: str | None,
) -> EngramShardTables:
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    # Resolve the checkpoint folder (local path, HF or ModelScope cache) the
    # way the attention ranks do.
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    hf_folder, _, use_safetensors, index_file = loader._prepare_weights(
        model, None, revision, False, None
    )
    if not use_safetensors:
        raise ValueError("Engram disaggregation needs a safetensors checkpoint")
    index_path = os.path.join(hf_folder, index_file)
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]

    weights, scales = [], []
    for layer_index, layer_id in enumerate(shard.layer_ids):
        start, end = ranges[layer_index]
        weight_name, scale_name = table_names(layer_id)
        for name in (weight_name, scale_name):
            if name not in weight_map:
                raise ValueError(f"{name} is missing from {index_path}")
        weight = _read_rows(hf_folder, weight_map[weight_name], weight_name, start, end)
        scale = _read_rows(hf_folder, weight_map[scale_name], scale_name, start, end)
        if weight.shape != (end - start, 256) or scale.shape != (
            end - start,
            SCALE_BYTES_PER_ROW,
        ):
            raise ValueError(
                f"{weight_name}: unexpected shapes {tuple(weight.shape)} / "
                f"{tuple(scale.shape)}"
            )
        full_rows = _tensor_rows(hf_folder, weight_map[weight_name], weight_name)
        if full_rows != shard.num_embeddings[layer_index]:
            raise ValueError(
                f"{weight_name} has {full_rows} rows, the config expects "
                f"{shard.num_embeddings[layer_index]}"
            )
        weights.append(weight)
        scales.append(scale)
    return EngramShardTables(weights, scales, [start for start, _ in ranges])


def _read_rows(
    hf_folder: str, file: str, name: str, start: int, end: int
) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(os.path.join(hf_folder, file), framework="pt") as handle:
        rows = handle.get_slice(name)[start:end]
    # E4M3 values and E8M0 scales are kept as their raw bytes.
    return rows.view(torch.uint8).contiguous()


def _tensor_rows(hf_folder: str, file: str, name: str) -> int:
    from safetensors import safe_open

    with safe_open(os.path.join(hf_folder, file), framework="pt") as handle:
        return handle.get_slice(name).get_shape()[0]
