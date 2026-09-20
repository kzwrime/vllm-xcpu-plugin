"""A-side vLLM support-matrix validation for the AF-EP MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttentionSupport:
    use_v2_model_runner: bool
    enable_expert_parallel: bool
    logical_ep_size: int
    eager: bool
    dtype: str
    quantization: str | None = None
    enable_dbo: bool = False
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    enable_eplb: bool = False
    fuse_shared_experts: bool = False
    enable_lora: bool = False
    speculative_configured: bool = False
    multimodal: bool = False


def support_from_vllm(vllm_config: Any) -> AttentionSupport:
    """Extract only the vLLM settings that constrain the AF-EP MVP."""
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    hf_config = model.hf_text_config
    num_experts = getattr(hf_config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "n_routed_experts", None)
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("AF-EP requires a routed MoE model")

    # Reject an active MM data path, not merely an MM-capable architecture.
    multimodal_config = getattr(model, "multimodal_config", None)
    multimodal = bool(
        multimodal_config is not None
        and not getattr(multimodal_config, "language_model_only", False)
    )

    import vllm.envs as vllm_envs

    return AttentionSupport(
        use_v2_model_runner=vllm_config.use_v2_model_runner,
        enable_expert_parallel=parallel.enable_expert_parallel,
        logical_ep_size=parallel.world_size_across_dp,
        eager=bool(model.enforce_eager),
        dtype=str(model.dtype),
        quantization=model.quantization,
        enable_dbo=parallel.enable_dbo,
        pipeline_parallel_size=parallel.pipeline_parallel_size,
        prefill_context_parallel_size=parallel.prefill_context_parallel_size,
        decode_context_parallel_size=parallel.decode_context_parallel_size,
        enable_eplb=parallel.enable_eplb,
        fuse_shared_experts=bool(vllm_envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS),
        enable_lora=vllm_config.lora_config is not None,
        speculative_configured=vllm_config.speculative_config is not None,
        multimodal=multimodal,
    )


def validate_attention_support(
    ep_size: int,
    support: AttentionSupport,
) -> None:
    failures: list[str] = []
    if not support.use_v2_model_runner:
        failures.append("ModelRunner V2 is required")
    if not support.enable_expert_parallel:
        failures.append("expert parallelism must be enabled")
    if ep_size <= 0:
        failures.append("at least one F rank is required")
    if not support.eager:
        failures.append("compile is not supported; eager execution is required")
    if support.dtype.lower() not in {"bf16", "bfloat16", "torch.bfloat16"}:
        failures.append(f"only BF16 is supported, got {support.dtype!r}")
    if support.quantization not in {None, "fp8", "compressed-tensors", "quark"}:
        failures.append(
            f"quantization {support.quantization!r} is not supported"
        )
    unsupported = {
        "DBO": support.enable_dbo,
        "pipeline parallelism": support.pipeline_parallel_size != 1,
        "prefill context parallelism": support.prefill_context_parallel_size != 1,
        "decode context parallelism": support.decode_context_parallel_size != 1,
        "EPLB": support.enable_eplb,
        "shared-expert fusion": support.fuse_shared_experts,
        "LoRA": support.enable_lora,
        "speculative decoding": support.speculative_configured,
        "multimodal input": support.multimodal,
    }
    failures.extend(
        f"{name} is not supported" for name, active in unsupported.items() if active
    )
    if failures:
        raise ValueError("invalid AF-EP configuration: " + "; ".join(failures))
