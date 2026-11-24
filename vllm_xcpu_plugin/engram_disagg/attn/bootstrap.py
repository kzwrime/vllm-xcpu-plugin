"""Model-worker bootstrap for the attention ranks of Engram disaggregation."""

from __future__ import annotations

from typing import Any

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld

from ..common import EngramSession
from .client import RemoteEngramClient
from .runtime import get_engram_client, register_engram_client

_SUPPORTED_BASE_LOAD_FORMATS = ("auto", "safetensors", "dummy")


def validate_attention_config(vllm_config: Any, num_engram_ranks: int) -> None:
    """Reject everything the exchange does not support, before model load."""
    failures: list[str] = []
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
    n_heads = getattr(model_config.hf_text_config, "engram_n_heads", 0)
    max_ngram = getattr(model_config.hf_text_config, "engram_max_ngram_size", 0)
    total_cols = (max_ngram - 1) * n_heads
    if num_engram_ranks <= 0:
        failures.append("at least one Engram rank is required")
    elif total_cols <= 0 or total_cols % num_engram_ranks:
        failures.append(
            f"{total_cols} engram hash columns cannot be split over "
            f"{num_engram_ranks} Engram ranks"
        )
    if parallel.pipeline_parallel_size != 1:
        failures.append("pipeline parallelism is not supported")
    if parallel.enable_dbo or getattr(parallel, "use_ubatching", False):
        failures.append("DBO / microbatching is not supported")
    engram_config = vllm_config.engram_config
    if engram_config is not None and engram_config.embedding_across_dp:
        failures.append("engram_config.embedding_across_dp is not supported")
    load_config = vllm_config.load_config
    base_format = load_config.load_format.removeprefix("xcpu_engram_")
    if base_format not in _SUPPORTED_BASE_LOAD_FORMATS:
        failures.append(f"load format {load_config.load_format!r} is not supported")
    extra = load_config.model_loader_extra_config or {}
    if extra.get("enable_multithread_load"):
        failures.append("enable_multithread_load is not supported")
    if extra.get("enable_weights_track"):
        failures.append("enable_weights_track is not supported (the tables are remote)")
    if load_config.safetensors_load_strategy not in (None, "lazy"):
        failures.append(
            f"safetensors_load_strategy={load_config.safetensors_load_strategy!r} "
            "is not supported"
        )
    if failures:
        raise ValueError(
            "invalid Engram disaggregation configuration: " + "; ".join(failures)
        )


def bootstrap_engram_client(
    *, mpi_world: MpiWorld, vllm_config: Any
) -> RemoteEngramClient:
    """Validate the worker and register its client before model construction."""
    if get_engram_client() is not None:
        raise RuntimeError("an Engram disaggregation client is already registered")
    assert mpi_world.cluster_type == ClusterType.ATTN
    session = EngramSession(
        mpi_world,
        max_tokens_proposal=vllm_config.scheduler_config.max_num_batched_tokens,
    )
    validate_attention_config(vllm_config, session.num_engram_ranks)
    client = RemoteEngramClient(session)
    register_engram_client(client)
    return client
