"""Attention-rank checkpoint loading that never reads the Engram tables.

The tables (``layers.N.engram.embed.{weight,scale}``, 94 GiB per layer) sit
in safetensors shards next to small tensors the attention rank needs
(``wkv``, ``q_weight``, ``k_weight``) and possibly MoE expert tensors, so
the shards cannot be dropped as a whole and the stock iterator would
materialize the tables before the model could decline them. This loader
walks those shards itself, skipping the two table names and applying
vLLM's expert-parallel filter (``should_skip_weight``) to everything else
they hold; every other shard goes through vLLM's iterator unchanged.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Generator
from dataclasses import replace

import torch
from safetensors import safe_open
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
    should_skip_weight,
)

_FORMAT_PREFIX = "xcpu_engram_"
_BASE_FORMATS = ("auto", "safetensors")
_TABLE_NAME = re.compile(r"(^|\.)engram\.embed\.(weight|scale)$")


def is_engram_table(name: str) -> bool:
    return _TABLE_NAME.search(name) is not None


def configure_attention_model_loader(load_config: LoadConfig) -> None:
    """Route the attention rank through the table-skipping loader."""
    load_format = load_config.load_format
    if load_format == "dummy":
        return
    base_format = load_format.removeprefix(_FORMAT_PREFIX)
    if base_format not in _BASE_FORMATS:
        raise ValueError(
            f"Engram disaggregation does not support load format {load_format!r}"
        )
    load_config.load_format = _FORMAT_PREFIX + base_format


def shards_with_engram_tables(hf_folder: str, index_file: str) -> set[str]:
    """Shard file names that hold an Engram table, from the safetensors index."""
    index_path = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_path):
        raise ValueError(
            f"Engram disaggregation needs the safetensors index {index_path} to "
            "locate the Engram tables"
        )
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]
    return {file for name, file in weight_map.items() if is_engram_table(name)}


def skip_engram_tables_iterator(
    files: list[str], local_expert_ids: set[int] | None = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Walk the shards that hold Engram tables: drop the tables, keep the
    neighbours under the same expert-parallel filter the stock iterator
    applies (foreign experts never reach this rank's loader)."""
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                if is_engram_table(name):
                    continue
                if should_skip_weight(name, local_expert_ids):
                    continue
                yield name, handle.get_tensor(name)


class EngramAttentionModelLoader(DefaultModelLoader):
    # Annotated for mypy: the base attribute is not visible to it here.
    counter_before_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        base_format = load_config.load_format.removeprefix(_FORMAT_PREFIX)
        if base_format not in _BASE_FORMATS:
            raise ValueError(f"unsupported Engram base load format: {base_format}")
        super().__init__(replace(load_config, load_format=base_format))

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        extra_config = self.load_config.model_loader_extra_config
        if extra_config.get("enable_multithread_load"):
            raise ValueError(
                "Engram disaggregation does not support enable_multithread_load"
            )
        if self.load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "Engram disaggregation does not support safetensors_load_strategy="
                f"{self.load_config.safetensors_load_strategy!r}"
            )
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
            raise ValueError("Engram disaggregation needs a safetensors checkpoint")
        table_shards = shards_with_engram_tables(hf_folder, index_file)
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
            yield from skip_engram_tables_iterator(with_tables, self.local_expert_ids)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        return ((source.prefix + name, tensor) for name, tensor in iterate())


for _base_format in _BASE_FORMATS:
    register_model_loader(_FORMAT_PREFIX + _base_format)(EngramAttentionModelLoader)
