# SPDX-License-Identifier: Apache-2.0

import os

import torch

_GUMBEL_SAMPLE_PATCHED = False
_TEMPERATURE_PATCHED = False


def _is_xcpu_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type in ("mcpu", "privateuseone")


def _xcpu_gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    import torch_xcpu

    # The x86 development backend may expose a no-op placeholder while the
    # platform implementation is supplied by the operator team.  A defined
    # fallback keeps warmup tokens in range; real implementations overwrite it.
    sampled_out = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
    return torch_xcpu.ops.gumbel_sample(
        logits=logits,
        sampled_out=sampled_out,
        expanded_idx_mapping=expanded_idx_mapping,
        temperature=temperature,
        seed=seed,
        pos=pos,
        apply_temperature=apply_temperature,
        output_processed_logits=output_processed_logits,
        output_processed_logits_col=output_processed_logits_col,
        use_fp64=use_fp64,
    )


def maybe_patch_vllm_temperature() -> None:
    """Keep excluded gumbel temperature handling on the platform op."""
    global _TEMPERATURE_PATCHED
    if _TEMPERATURE_PATCHED:
        return

    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module
    import vllm.v1.worker.gpu.sample.states as states_module

    original = gumbel_module.apply_temperature

    def apply_temperature(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        if not _is_xcpu_device(logits):
            return original(logits, expanded_idx_mapping, temperature)
        torch.ops.mcpu.vllm_temperature_kernel(
            logits,
            expanded_idx_mapping,
            temperature,
            logits.shape[1],
        )

    gumbel_module.apply_temperature = apply_temperature
    states_module.apply_temperature = apply_temperature
    _TEMPERATURE_PATCHED = True


def maybe_patch_vllm_gumbel_sample() -> None:
    global _GUMBEL_SAMPLE_PATCHED

    if not bool(int(os.getenv("VLLM_USE_XCPU_GUMBEL_SAMPLE", "1"))):
        return
    if _GUMBEL_SAMPLE_PATCHED:
        return

    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module

    original_gumbel_sample = gumbel_module.gumbel_sample

    def _patched_gumbel_sample(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        apply_temperature: bool,
        output_processed_logits: torch.Tensor | None = None,
        output_processed_logits_col: torch.Tensor | None = None,
        use_fp64: bool = False,
    ) -> torch.Tensor:
        if _is_xcpu_device(logits):
            return _xcpu_gumbel_sample(
                logits,
                expanded_idx_mapping,
                temperature,
                seed,
                pos,
                apply_temperature,
                output_processed_logits,
                output_processed_logits_col,
                use_fp64,
            )

        return original_gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            output_processed_logits,
            output_processed_logits_col,
            use_fp64,
        )

    gumbel_module.gumbel_sample = _patched_gumbel_sample

    # If sampler.py was imported before plugin registration, its direct import
    # has already captured the original function object. Patch that binding too.
    try:
        import vllm.v1.worker.gpu.sample.sampler as sampler_module

        sampler_module.gumbel_sample = _patched_gumbel_sample
    except ImportError:
        pass

    _GUMBEL_SAMPLE_PATCHED = True
