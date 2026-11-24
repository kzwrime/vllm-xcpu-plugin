"""Checkpoint loader for EngramParallel: slices each rank's rows on read."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Generator, Mapping
from dataclasses import replace

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

from vllm_xcpu_plugin.engram_disagg.attn.model_loader import (
    is_engram_table,
    shards_with_engram_tables,
    skip_engram_tables_iterator,
)
from vllm_xcpu_plugin.engram_disagg.attn.runtime import get_engram_client

_FORMAT_PREFIX = "xcpu_engram_parallel_"
_BASE_FORMATS = ("auto", "safetensors")
_TABLE_NAME_WITH_LAYER = re.compile(
    r"(?:^|\.)layers\.(\d+)\.engram\.embed\.(weight|scale)$"
)


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


def slice_engram_tables_iterator(
    files: list[str],
    layer_slices: Mapping[int, tuple[int, int, int]],
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Walk shards containing Engram tables: slice each rank's ``[lo:hi]`` rows
    from ``engram.embed.{weight,scale}`` and apply ``should_skip_weight`` to
    neighbouring tensors."""
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                match = _TABLE_NAME_WITH_LAYER.search(name)
                if match is not None:
                    layer_id = int(match.group(1))
                    if layer_id not in layer_slices:
                        raise ValueError(
                            f"Engram table {name} (layer {layer_id}) missing from "
                            "EngramParallelLayout"
                        )
                    lo, hi, expected_rows = layer_slices[layer_id]
                    tensor_slice = handle.get_slice(name)
                    full_rows = tensor_slice.get_shape()[0]
                    if full_rows != expected_rows:
                        raise ValueError(
                            f"{name} has {full_rows} rows, expected {expected_rows}"
                        )
                    yield name, tensor_slice[lo:hi].contiguous()
                    continue
                if is_engram_table(name):
                    raise ValueError(
                        f"unexpected Engram table tensor name {name!r} without "
                        "layers.<id>.engram.embed.<kind>"
                    )
                if should_skip_weight(name, local_expert_ids):
                    continue
                yield name, handle.get_tensor(name)


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
        table_shards = shards_with_engram_tables(
            hf_folder,
            index_file,
            hf_weights_files=hf_weights_files,
            mode_name="EngramParallel",
        )
        with_tables = [
            f for f in hf_weights_files if os.path.basename(f) in table_shards
        ]
        without_tables = [
            f for f in hf_weights_files if os.path.basename(f) not in table_shards
        ]
        if len(with_tables) != len(table_shards):
            raise ValueError(
                f"Engram table shards {sorted(table_shards)} are not all among the "
                "checkpoint files to load"
            )
        if not self._skip_engram_tables and not table_shards:
            raise ValueError(
                "EngramParallel expected Engram table tensors in the checkpoint, "
                "but none were found"
            )

        skip_tables = self._skip_engram_tables

        def iterate():
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
            if with_tables:
                if skip_tables:
                    yield from skip_engram_tables_iterator(
                        with_tables, self.local_expert_ids
                    )
                else:
                    assert layer_slices is not None
                    yield from slice_engram_tables_iterator(
                        with_tables, layer_slices, self.local_expert_ids
                    )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        return ((source.prefix + name, tensor) for name, tensor in iterate())


for _base_format in _BASE_FORMATS:
    register_model_loader(_FORMAT_PREFIX + _base_format)(EngramParallelModelLoader)
