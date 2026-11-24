# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 platform config rules on MCPU.

Called from ``McpuPlatform.check_and_update_config``. XCPU stores the SM100
record set (see ``attention.py``), so the indexer K cache is always the 68 B
MXFP4 record: an unset ``indexer_kv_dtype`` resolves to it and any other
choice fails here, before a layer is built with a cache it cannot read.
"""

from vllm.config import VllmConfig
from vllm.logger import init_logger

import vllm_xcpu_plugin.envs as envs_xcpu

logger = init_logger(__name__)


def update_deepseek_v41_config(vllm_config: VllmConfig) -> None:
    engram_config = getattr(vllm_config, "engram_config", None)
    if engram_config is not None and engram_config.cpu_offload:
        # MCPU device memory already is host memory and has no UVA view:
        # keep the n-gram tables as ordinary device tensors.
        engram_config.cpu_offload = False
        engram_config.dp_shared_memory = False
        engram_config.use_thp = False
        logger.info("Engram tables stay in MCPU memory (cpu_offload=False).")

    model_config = vllm_config.model_config
    model_type = getattr(getattr(model_config, "hf_config", None), "model_type", "")
    if (
        envs_xcpu.VLLM_XCPU_ENABLE_ENGRAM_DISAGG
        and envs_xcpu.VLLM_XCPU_ENABLE_ENGRAM_PARALLEL
    ):
        raise ValueError(
            "VLLM_XCPU_ENABLE_ENGRAM_DISAGG and VLLM_XCPU_ENABLE_ENGRAM_PARALLEL "
            "cannot both be enabled"
        )
    if envs_xcpu.VLLM_XCPU_ENABLE_ENGRAM_DISAGG:
        _update_engram_disagg_config(vllm_config, engram_config, str(model_type))
    elif envs_xcpu.VLLM_XCPU_ENABLE_ENGRAM_PARALLEL:
        _update_engram_parallel_config(vllm_config, engram_config, str(model_type))
    if not str(model_type).startswith("deepseek_v41"):
        return
    attention_config = vllm_config.attention_config
    indexer_kv_dtype = attention_config.resolve_indexer_kv_dtype("mxfp4")
    if indexer_kv_dtype != "mxfp4":
        raise ValueError(
            f"indexer_kv_dtype={indexer_kv_dtype!r}: the XCPU DeepSeek-V4.1 "
            "indexer stores the MXFP4 key record only; use 'mxfp4' or leave it "
            "unset."
        )
    if attention_config.indexer_kv_dtype != "mxfp4":
        attention_config.indexer_kv_dtype = "mxfp4"
        logger.info("DeepSeek-V4.1 indexer K cache: mxfp4 (the XCPU record).")
    if attention_config.indexer_sparse_logits:
        raise ValueError(
            "indexer_sparse_logits selects the CUDA sparse MQA-logits indexer; "
            "the XCPU indexer scores candidate blocks in its own kernel. Leave "
            "it unset."
        )


def _update_distributed_engram_config(
    vllm_config: VllmConfig,
    engram_config,
    model_type: str,
    *,
    mode_name: str,
    env_flag: str,
    world_desc: str,
    table_location: str,
    configure_loader,
) -> None:
    """Shared validation and knob overrides for distributed Engram modes."""
    if not model_type.startswith("deepseek_v41") or engram_config is None:
        raise ValueError(
            f"{env_flag} needs a DeepSeek-V4.1 model with engram "
            f"layers, got model_type={model_type!r}"
        )
    if not envs_xcpu.VLLM_CPU_USE_MPI:
        raise ValueError(
            f"{env_flag} requires the MPI launcher "
            f"(VLLM_CPU_USE_MPI=1): {world_desc}"
        )
    if engram_config.embedding_across_dp:
        raise ValueError(
            f"engram_config.embedding_across_dp is not supported with {mode_name}; "
            f"{table_location}"
        )
    overridden = [
        name
        for name in ("cpu_offload", "dp_shared_memory", "use_thp")
        if getattr(engram_config, name)
    ]
    if overridden:
        logger.info(
            "%s: %s; ignoring engram_config %s.",
            mode_name,
            table_location,
            ", ".join(overridden),
        )
    engram_config.cpu_offload = False
    engram_config.dp_shared_memory = False
    engram_config.use_thp = False

    configure_loader(vllm_config.load_config)
    logger.info(
        "%s: %s (load format %s).",
        mode_name,
        table_location,
        vllm_config.load_config.load_format,
    )


def _update_engram_disagg_config(
    vllm_config: VllmConfig, engram_config, model_type: str
) -> None:
    """Engram disaggregation: the attention rank keeps no table."""
    from vllm_xcpu_plugin.engram_disagg.attn.model_loader import (
        configure_attention_model_loader,
    )

    _update_distributed_engram_config(
        vllm_config,
        engram_config,
        model_type,
        mode_name="Engram disaggregation",
        env_flag="VLLM_XCPU_ENABLE_ENGRAM_DISAGG",
        world_desc="the Engram ranks join the MPMD world",
        table_location="the tables live on the Engram ranks",
        configure_loader=configure_attention_model_loader,
    )


def _update_engram_parallel_config(
    vllm_config: VllmConfig, engram_config, model_type: str
) -> None:
    """EngramParallel: all DP x TP ranks shard the tables in 2D."""
    from vllm_xcpu_plugin.engram_parallel.model_loader import (
        configure_engram_parallel_model_loader,
    )

    _update_distributed_engram_config(
        vllm_config,
        engram_config,
        model_type,
        mode_name="EngramParallel",
        env_flag="VLLM_XCPU_ENABLE_ENGRAM_PARALLEL",
        world_desc="EngramParallel communicates across the MPI worker world",
        table_location="the tables are 2D-sharded across all worker ranks",
        configure_loader=configure_engram_parallel_model_loader,
    )
