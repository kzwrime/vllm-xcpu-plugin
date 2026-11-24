"""Worker bootstrap for EngramParallel across all DP x TP ranks."""

from __future__ import annotations

from typing import Any

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld
from vllm_xcpu_plugin.engram_disagg.attn.runtime import (
    get_engram_client,
    register_engram_client,
)

from .client import EngramParallelClient
from .layout import EngramParallelLayout
from .session import EngramParallelSession

_SUPPORTED_BASE_LOAD_FORMATS = ("auto", "safetensors", "dummy")


def validate_engram_parallel_config(vllm_config: Any, world_size: int) -> None:
    """Reject unsupported configurations before model construction."""
    failures: list[str] = []
    if envs_xcpu.VLLM_XCPU_ENABLE_ENGRAM_DISAGG:
        failures.append(
            "VLLM_XCPU_ENABLE_ENGRAM_PARALLEL cannot be combined with "
            "VLLM_XCPU_ENABLE_ENGRAM_DISAGG"
        )
    model_config = vllm_config.model_config
    parallel = vllm_config.parallel_config
    if model_config.architecture != "DeepseekV41ForCausalLM":
        failures.append(
            "only DeepseekV41ForCausalLM is supported, got "
            f"{model_config.architecture!r}"
        )
    layer_ids = getattr(model_config.hf_text_config, "engram_layer_ids", None)
    if not layer_ids:
        failures.append("the model has no engram layers")
    if world_size <= 0:
        failures.append("world_size must be positive")
    if parallel.pipeline_parallel_size != 1:
        failures.append("pipeline parallelism is not supported")
    if getattr(parallel, "prefill_context_parallel_size", 1) != 1:
        failures.append("prefill context parallelism is not supported")
    if getattr(parallel, "decode_context_parallel_size", 1) != 1:
        failures.append("decode context parallelism is not supported")
    if getattr(parallel, "enable_elastic_ep", False):
        failures.append("elastic EP is not supported")
    if parallel.enable_dbo or getattr(parallel, "use_ubatching", False):
        failures.append("DBO / microbatching is not supported")
    dp_size = getattr(parallel, "data_parallel_size", 1)
    dp_rank = getattr(parallel, "data_parallel_rank", 0)
    if dp_size <= 0 or not (0 <= dp_rank < dp_size):
        failures.append(
            f"invalid DP topology data_parallel_rank={dp_rank}, "
            f"data_parallel_size={dp_size}"
        )
    if hasattr(parallel, "tensor_parallel_size"):
        tp_size = parallel.tensor_parallel_size
        if world_size > 0 and dp_size > 0 and dp_size * tp_size != world_size:
            failures.append(
                f"dp_size ({dp_size}) * tp_size ({tp_size}) "
                f"!= world_size ({world_size})"
            )
    engram_config = vllm_config.engram_config
    if engram_config is not None and engram_config.embedding_across_dp:
        failures.append("engram_config.embedding_across_dp is not supported")
    load_config = vllm_config.load_config
    base_format = load_config.load_format.removeprefix("xcpu_engram_parallel_")
    if base_format not in _SUPPORTED_BASE_LOAD_FORMATS:
        failures.append(f"load format {load_config.load_format!r} is not supported")
    extra = load_config.model_loader_extra_config or {}
    if extra.get("enable_multithread_load"):
        failures.append("enable_multithread_load is not supported")
    if load_config.safetensors_load_strategy not in (None, "lazy"):
        failures.append(
            f"safetensors_load_strategy={load_config.safetensors_load_strategy!r} "
            "is not supported"
        )
    if failures:
        raise ValueError(
            "invalid EngramParallel configuration: " + "; ".join(failures)
        )


def bootstrap_engram_parallel_client(
    *,
    mpi_world: MpiWorld,
    vllm_config: Any,
    world_rank: int,
    world_size: int,
) -> EngramParallelClient:
    """Validate worker config, build layout, and register EngramParallelClient."""
    if get_engram_client() is not None:
        raise RuntimeError("an Engram client is already registered")
    if mpi_world.cluster_type != ClusterType.ATTN:
        raise ValueError("EngramParallel requires ClusterType.ATTN worker ranks")
    validate_engram_parallel_config(vllm_config, world_size)
    layout = EngramParallelLayout.from_hf_text_config(
        vllm_config.model_config.hf_text_config, world_size
    )
    session = EngramParallelSession(
        mpi_world,
        world_rank=world_rank,
        world_size=world_size,
        max_tokens_proposal=vllm_config.scheduler_config.max_num_batched_tokens,
    )
    parallel = vllm_config.parallel_config
    client = EngramParallelClient(
        session,
        layout,
        dp_size=getattr(parallel, "data_parallel_size", 1),
        dp_rank=getattr(parallel, "data_parallel_rank", 0),
    )
    register_engram_client(client)
    return client
