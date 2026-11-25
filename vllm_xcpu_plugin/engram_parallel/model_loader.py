"""Checkpoint loader for EngramParallel: slices each rank's rows on read."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
    should_skip_weight,
)
from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

from vllm_xcpu_plugin.engram_disagg.attn.runtime import get_engram_client

_FORMAT_PREFIX = "xcpu_engram_parallel_"
_BASE_FORMATS = ("auto", "safetensors")
_UNSPLIT_TABLE_RE = re.compile(
    r"^(?P<base>(?:.+\.)?layers\.(?P<layer>\d+)\.engram\.embed\.(?P<kind>weight|scale))$"
)
_SPLIT_TABLE_RE = re.compile(
    r"^(?P<base>(?:.+\.)?layers\.(?P<layer>\d+)\.engram\.embed\.(?P<kind>weight|scale))"
    r"\.row_slice\.(?P<start>\d+)_(?P<end>\d+)$"
)
_ANY_ENGRAM_TABLE_RE = re.compile(r"(?:^|\.)engram\.embed\.(?:weight|scale)(?:$|\.)")

_STR_TO_DTYPE = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "F8_E4M3": torch.float8_e4m3fn,
    "float8_e8m0fnu": torch.float8_e8m0fnu,
    "F8_E8M0": torch.float8_e8m0fnu,
    "uint8": torch.uint8,
    "U8": torch.uint8,
    "bfloat16": torch.bfloat16,
    "BF16": torch.bfloat16,
    "float16": torch.float16,
    "F16": torch.float16,
    "float32": torch.float32,
    "F32": torch.float32,
}


@dataclass(frozen=True)
class _EngramRowChunk:
    tensor_key: str
    shard: str
    row_start: int
    row_end: int


@dataclass(frozen=True)
class _EngramTablePlan:
    base_name: str
    layer_id: int
    kind: str
    unsplit_shard: str | None = None
    chunks: tuple[_EngramRowChunk, ...] = ()

    @property
    def is_split(self) -> bool:
        return self.unsplit_shard is None

    @property
    def total_rows(self) -> int:
        assert self.chunks
        return self.chunks[-1].row_end


def configure_engram_parallel_model_loader(load_config: LoadConfig) -> None:
    """Route model loading through the table-slicing loader."""
    load_format = load_config.load_format
    if load_format == "dummy":
        return
    base_format = load_format.removeprefix(_FORMAT_PREFIX)
    if base_format not in _BASE_FORMATS:
        raise ValueError(
            f"EngramParallel does not support load format {load_format!r}"
        )
    load_config.load_format = _FORMAT_PREFIX + base_format


def _build_table_plans(
    weight_map: Mapping[str, str],
    slice_meta: Mapping[str, Any],
) -> tuple[dict[str, _EngramTablePlan], set[str], set[str]]:
    shards_with_table_keys: set[str] = set()
    shards_with_regular_keys: set[str] = set()
    unsplit: dict[str, tuple[int, str, str]] = {}
    split_chunks: dict[str, tuple[int, str, list[_EngramRowChunk]]] = {}

    for name, file in weight_map.items():
        shard = os.path.basename(file)
        m_split = _SPLIT_TABLE_RE.match(name)
        if m_split is not None:
            base = m_split.group("base")
            layer_id = int(m_split.group("layer"))
            kind = m_split.group("kind")
            start = int(m_split.group("start"))
            end = int(m_split.group("end"))
            if start >= end:
                raise ValueError(
                    f"invalid Engram row slice [{start}, {end}) in {name!r}"
                )
            shards_with_table_keys.add(shard)
            entry = split_chunks.get(base)
            if entry is None:
                entry = (layer_id, kind, [])
                split_chunks[base] = entry
            entry[2].append(_EngramRowChunk(name, shard, start, end))
            continue

        m_unsplit = _UNSPLIT_TABLE_RE.match(name)
        if m_unsplit is not None:
            base = m_unsplit.group("base")
            layer_id = int(m_unsplit.group("layer"))
            kind = m_unsplit.group("kind")
            shards_with_table_keys.add(shard)
            if base in unsplit:
                raise ValueError(f"duplicate Engram table key {name!r}")
            unsplit[base] = (layer_id, kind, shard)
            continue

        if _ANY_ENGRAM_TABLE_RE.search(name):
            raise ValueError(
                f"unexpected Engram table tensor name {name!r} without "
                "layers.<id>.engram.embed.<kind>"
            )

        shards_with_regular_keys.add(shard)

    mixed = set(unsplit) & set(split_chunks)
    if mixed:
        raise ValueError(
            f"Engram table(s) {sorted(mixed)} mix split and unsplit tensors"
        )

    plans: dict[str, _EngramTablePlan] = {}
    for base, (layer_id, kind, shard) in unsplit.items():
        plans[base] = _EngramTablePlan(
            base_name=base,
            layer_id=layer_id,
            kind=kind,
            unsplit_shard=shard,
        )

    for base, (layer_id, kind, chunks) in split_chunks.items():
        chunks.sort(key=lambda c: c.row_start)
        if chunks[0].row_start != 0:
            raise ValueError(
                f"Engram row slices for {base!r} start at {chunks[0].row_start}, "
                "expected 0"
            )
        for prev, curr in zip(chunks, chunks[1:]):
            if prev.row_end < curr.row_start:
                raise ValueError(
                    f"Engram row slices for {base!r} have a gap between "
                    f"[{prev.row_start}, {prev.row_end}) and "
                    f"[{curr.row_start}, {curr.row_end})"
                )
            if prev.row_end > curr.row_start:
                raise ValueError(
                    f"Engram row slices for {base!r} overlap between "
                    f"[{prev.row_start}, {prev.row_end}) and "
                    f"[{curr.row_start}, {curr.row_end})"
                )
        meta = slice_meta.get(base)
        if isinstance(meta, Mapping) and "shape" in meta:
            shape = meta["shape"]
            if (
                not isinstance(shape, list)
                or not shape
                or int(shape[0]) != chunks[-1].row_end
            ):
                raise ValueError(
                    f"Engram metadata shape mismatch for {base!r}: "
                    f"metadata={shape!r}, slices end at {chunks[-1].row_end}"
                )
        plans[base] = _EngramTablePlan(
            base_name=base,
            layer_id=layer_id,
            kind=kind,
            chunks=tuple(chunks),
        )

    return plans, shards_with_table_keys, shards_with_regular_keys


def _inspect_engram_checkpoint(
    hf_folder: str,
    index_file: str,
    hf_weights_files: list[str],
) -> tuple[dict[str, _EngramTablePlan], set[str], set[str], Mapping[str, Any]]:
    index_path = os.path.join(hf_folder, index_file) if index_file else ""
    if index_path and os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index_data = json.load(handle)
        weight_map = index_data["weight_map"]
        meta = index_data.get("metadata", {})
        slice_meta = (
            meta.get("engram_row_slices", {}) if isinstance(meta, Mapping) else {}
        )
        if not isinstance(slice_meta, Mapping):
            slice_meta = {}
        plans, table_shards, regular_shards = _build_table_plans(weight_map, slice_meta)
        return plans, table_shards, regular_shards, slice_meta

    if len(hf_weights_files) == 1:
        single_file = hf_weights_files[0]
        basename = os.path.basename(single_file)
        with safe_open(single_file, framework="pt") as handle:
            weight_map = {name: basename for name in handle.keys()}  # noqa: SIM118
        del handle
        plans, table_shards, regular_shards = _build_table_plans(weight_map, {})
        return plans, table_shards, regular_shards, {}

    raise ValueError(
        f"EngramParallel needs the safetensors index {index_path} to locate the "
        "Engram tables"
    )


def _skip_engram_tables_iterator(
    files: list[str],
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield non-Engram tensors from shards that also hold Engram tables,
    cloning each kept tensor so the file mmap can be released immediately."""
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                if _ANY_ENGRAM_TABLE_RE.search(name):
                    continue
                if should_skip_weight(name, local_expert_ids):
                    continue
                yield name, handle.get_tensor(name).clone()
        del handle


def _load_engram_table_slice(
    plan: _EngramTablePlan,
    lo: int,
    hi: int,
    expected_rows: int,
    files_by_basename: Mapping[str, str],
    slice_meta: Mapping[str, Any],
) -> torch.Tensor:
    if lo < 0 or hi < lo or hi > expected_rows:
        raise ValueError(
            f"invalid Engram slice [{lo}, {hi}) for {plan.base_name!r} "
            f"(expected_rows={expected_rows})"
        )

    if not plan.is_split:
        assert plan.unsplit_shard is not None
        with safe_open(files_by_basename[plan.unsplit_shard], framework="pt") as handle:
            tensor_slice = handle.get_slice(plan.base_name)
            full_rows = tensor_slice.get_shape()[0]
            if full_rows != expected_rows:
                raise ValueError(
                    f"{plan.base_name} has {full_rows} rows, expected {expected_rows}"
                )
            out = tensor_slice[lo:hi].clone()
            del tensor_slice
        del handle
        return out

    if plan.total_rows != expected_rows:
        raise ValueError(
            f"{plan.base_name} has {plan.total_rows} rows, expected {expected_rows}"
        )

    if lo == hi:
        meta = slice_meta.get(plan.base_name)
        if isinstance(meta, Mapping) and "shape" in meta and "dtype" in meta:
            shape = meta["shape"]
            dtype = _STR_TO_DTYPE.get(str(meta["dtype"]))
            if isinstance(shape, list) and len(shape) >= 2 and dtype is not None:
                return torch.empty((0, *(int(d) for d in shape[1:])), dtype=dtype)
        first = plan.chunks[0]
        with safe_open(files_by_basename[first.shard], framework="pt") as handle:
            ts = handle.get_slice(first.tensor_key)
            c_shape = tuple(ts.get_shape())
            if len(c_shape) < 2 or c_shape[0] != first.row_end - first.row_start:
                raise ValueError(
                    f"{first.tensor_key} has shape {c_shape}, expected "
                    f"({first.row_end - first.row_start}, ...)"
                )
            out = ts[0:0].clone()
            del ts
        del handle
        return out

    overlapping = [c for c in plan.chunks if c.row_end > lo and c.row_start < hi]
    out_raw: torch.Tensor | None = None
    orig_dtype: torch.dtype | None = None
    tail_shape: tuple[int, ...] = ()

    for chunk in overlapping:
        ov_lo = max(lo, chunk.row_start)
        ov_hi = min(hi, chunk.row_end)
        with safe_open(files_by_basename[chunk.shard], framework="pt") as handle:
            ts = handle.get_slice(chunk.tensor_key)
            c_shape = tuple(ts.get_shape())
            if len(c_shape) < 2 or c_shape[0] != chunk.row_end - chunk.row_start:
                raise ValueError(
                    f"{chunk.tensor_key} has shape {c_shape}, expected "
                    f"({chunk.row_end - chunk.row_start}, ...)"
                )
            sub = ts[ov_lo - chunk.row_start : ov_hi - chunk.row_start]
            if out_raw is None:
                orig_dtype = sub.dtype
                tail_shape = tuple(int(d) for d in c_shape[1:])
                out_raw = torch.empty((hi - lo, *tail_shape), dtype=torch.uint8)
            elif (
                sub.dtype != orig_dtype
                or tuple(int(d) for d in c_shape[1:]) != tail_shape
            ):
                raise ValueError(
                    f"{chunk.tensor_key} shape/dtype {c_shape}/{sub.dtype} "
                    f"mismatch with previous chunks ({tail_shape}/{orig_dtype})"
                )
            out_raw[ov_lo - lo : ov_hi - lo].copy_(sub.view(torch.uint8))
            del sub, ts
        del handle

    assert out_raw is not None and orig_dtype is not None
    return out_raw.view(orig_dtype)


def _validate_and_iter_table_slices(
    table_plans: Mapping[str, _EngramTablePlan],
    layer_slices: Mapping[int, tuple[int, int, int]],
    files_by_basename: Mapping[str, str],
    slice_meta: Mapping[str, Any],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    found_kinds: dict[int, set[str]] = {}
    for plan in table_plans.values():
        if plan.layer_id not in layer_slices:
            raise ValueError(
                f"Engram table {plan.base_name} (layer {plan.layer_id}) missing from "
                "EngramParallelLayout"
            )
        found_kinds.setdefault(plan.layer_id, set()).add(plan.kind)

    ordered_plans = sorted(
        table_plans.values(),
        key=lambda p: (p.layer_id, 0 if p.kind == "weight" else 1, p.base_name),
    )
    for plan in ordered_plans:
        lo, hi, expected_rows = layer_slices[plan.layer_id]
        yield (
            plan.base_name,
            _load_engram_table_slice(
                plan,
                lo,
                hi,
                expected_rows,
                files_by_basename,
                slice_meta,
            ),
        )


def slice_engram_tables_iterator(
    files: list[str],
    layer_slices: Mapping[int, tuple[int, int, int]],
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Walk shards containing Engram tables (unsplit or pre-split): slice each
    rank's ``[lo:hi]`` rows and apply ``should_skip_weight`` to neighbours."""
    files_by_basename: dict[str, str] = {}
    weight_map: dict[str, str] = {}
    for file in files:
        basename = os.path.basename(file)
        if basename in files_by_basename:
            raise ValueError(f"duplicate shard filename {basename!r}")
        files_by_basename[basename] = file
        with safe_open(file, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                if name in weight_map:
                    raise ValueError(f"duplicate tensor name {name!r}")
                weight_map[name] = basename
        del handle

    table_plans, table_shards, regular_shards = _build_table_plans(weight_map, {})
    for plan in table_plans.values():
        if plan.layer_id not in layer_slices:
            raise ValueError(
                f"Engram table {plan.base_name} (layer {plan.layer_id}) missing from "
                "EngramParallelLayout"
            )

    regular_files = [
        file
        for file in files
        if os.path.basename(file) in regular_shards
        or os.path.basename(file) not in table_shards
    ]
    yield from _skip_engram_tables_iterator(regular_files, local_expert_ids)
    yield from _validate_and_iter_table_slices(
        table_plans,
        layer_slices,
        files_by_basename,
        {},
    )


class EngramParallelModelLoader(DefaultModelLoader):
    counter_before_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        base_format = load_config.load_format.removeprefix(_FORMAT_PREFIX)
        if base_format not in _BASE_FORMATS:
            raise ValueError(
                f"unsupported EngramParallel base load format: {base_format}"
            )
        super().__init__(replace(load_config, load_format=base_format))
        self._skip_engram_tables: bool = False

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        prev_skip = self._skip_engram_tables
        self._skip_engram_tables = not any(
            isinstance(module, ParallelEngramEmbedding) for module in model.modules()
        )
        try:
            yield from super().get_all_weights(model_config, model)
        finally:
            self._skip_engram_tables = prev_skip

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        extra_config = self.load_config.model_loader_extra_config
        if extra_config.get("enable_multithread_load"):
            raise ValueError(
                "EngramParallel does not support enable_multithread_load"
            )
        if self.load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "EngramParallel does not support safetensors_load_strategy="
                f"{self.load_config.safetensors_load_strategy!r}"
            )

        layer_slices: Mapping[int, tuple[int, int, int]] | None = None
        if not self._skip_engram_tables:
            client = get_engram_client()
            layout = getattr(client, "layout", None)
            session = getattr(client, "session", None)
            if layout is None or session is None or not hasattr(session, "world_rank"):
                raise RuntimeError(
                    "EngramParallelModelLoader requires an active EngramParallelClient"
                )
            layer_slices = {
                layer_id: layout.layer_slice_by_id(layer_id, session.world_rank)
                for layer_id in layout.layer_ids
            }

        hf_folder, hf_weights_files, use_safetensors, index_file = (
            self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
        )
        if not use_safetensors:
            raise ValueError("EngramParallel needs a safetensors checkpoint")

        table_plans, table_shards, regular_shards, slice_meta = (
            _inspect_engram_checkpoint(hf_folder, index_file, hf_weights_files)
        )
        files_by_basename = {os.path.basename(f): f for f in hf_weights_files}
        missing_shards = sorted(s for s in table_shards if s not in files_by_basename)
        if missing_shards:
            raise ValueError(
                f"Engram table shards {missing_shards} are not all among the "
                "checkpoint files to load"
            )

        without_tables = [
            f for f in hf_weights_files if os.path.basename(f) not in table_shards
        ]
        mixed_shards = [
            f
            for f in hf_weights_files
            if os.path.basename(f) in table_shards
            and os.path.basename(f) in regular_shards
        ]

        if not self._skip_engram_tables:
            if not table_plans:
                raise ValueError(
                    "EngramParallel expected Engram table tensors in the checkpoint, "
                    "but none were found"
                )
            assert layer_slices is not None
            for plan in table_plans.values():
                if plan.layer_id not in layer_slices:
                    raise ValueError(
                        f"Engram table {plan.base_name} (layer {plan.layer_id}) "
                        "missing from EngramParallelLayout"
                    )

        skip_tables = self._skip_engram_tables

        def iterate():
            if without_tables:
                yield from safetensors_weights_iterator(
                    without_tables,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.safetensors_load_strategy,
                    local_expert_ids=self.local_expert_ids,
                    safetensors_prefetch_num_threads=(
                        self.load_config.safetensors_prefetch_num_threads
                    ),
                    safetensors_prefetch_block_size=(
                        self.load_config.safetensors_prefetch_block_size
                    ),
                )
            if mixed_shards:
                yield from _skip_engram_tables_iterator(
                    mixed_shards, self.local_expert_ids
                )
            if not skip_tables:
                assert layer_slices is not None
                yield from _validate_and_iter_table_slices(
                    table_plans,
                    layer_slices,
                    files_by_basename,
                    slice_meta,
                )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        return ((source.prefix + name, tensor) for name, tensor in iterate())


for _base_format in _BASE_FORMATS:
    register_model_loader(_FORMAT_PREFIX + _base_format)(EngramParallelModelLoader)
