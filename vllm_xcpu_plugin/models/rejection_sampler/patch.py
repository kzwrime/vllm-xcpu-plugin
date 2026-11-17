# SPDX-License-Identifier: Apache-2.0
"""Fused XCPU rejection sampler patch for vLLM RejectionSampler._verify."""

from __future__ import annotations

import numpy as np
import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility
from vllm_xcpu_plugin.upstream_contracts import rebind

_INSTALLED = False

_SUPPORTED_LOGITS_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _can_fuse_target_temperature(sampler: Sampler, idx_mapping_np: np.ndarray) -> bool:
    """Return True iff target sampling params reduce strictly to temperature scaling."""
    if sampler.return_sampling_mask:
        return False
    if (
        sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        and sampler.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
    ):
        return False

    ss = sampler.sampling_states
    if not (
        np.all(ss.min_p.np[idx_mapping_np] == 0.0)
        and np.all(ss.top_k.np[idx_mapping_np] == ss.vocab_size)
        and np.all(ss.top_p.np[idx_mapping_np] == 1.0)
    ):
        return False

    tb = sampler.thinking_budget_state
    if tb.enabled and np.any(tb.use_thinking_budget[idx_mapping_np]):
        return False

    for proc in sampler.logits_processors:
        if isinstance(proc, LogitBiasState):
            if np.any(proc.use_logit_bias[idx_mapping_np]):
                return False
        elif isinstance(proc, PenaltiesState):
            if np.any(proc.use_penalty[idx_mapping_np]):
                return False
        elif isinstance(proc, BadWordsState):
            if not np.all(proc.num_bad_words.np[idx_mapping_np] == 0):
                return False
        else:
            return False

    return True


def _xcpu_rejection_sampler_verify(
    self: RejectionSampler,
    logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    seq_lens_upper_bound_np: np.ndarray,
    verify_draft_sampled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_eligible = (
        logits.device.type in ("mcpu", "privateuseone")
        and logits.ndim == 2
        and logits.stride(-1) == 1
        and logits.dtype in _SUPPORTED_LOGITS_DTYPES
        and (
            draft_logits is None
            or (
                draft_logits.ndim == 3
                and draft_logits.stride(-1) == 1
                and draft_logits.dtype in _SUPPORTED_LOGITS_DTYPES
            )
        )
        and self.synthetic_conditional_rates is None
        and not self.use_block_verification
        and self.watermark_key is None
    )
    if not fused_eligible:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            verify_draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        return processed_logits, sampled, num_sampled

    if not np.any(self.sampler.needs_logits_processing[idx_mapping_np]):
        processed_logits = logits
        apply_target_temperature = False
    elif _can_fuse_target_temperature(self.sampler, idx_mapping_np):
        processed_logits = logits
        apply_target_temperature = True
    else:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        apply_target_temperature = False

    from torch_xcpu import ops as xcpu_ops

    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = (
        min(processed_logits.shape[1], draft_logits.shape[-1])
        if draft_logits is not None
        else processed_logits.shape[1]
    )
    sampled = verify_draft_sampled.new_empty(
        num_reqs, self.num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    xcpu_ops.vllm_rejection_sample_fused(
        sampled,
        num_sampled,
        processed_logits,
        draft_logits,
        cu_num_logits,
        expanded_idx_mapping,
        verify_draft_sampled,
        self.sampler.sampling_states.temperature.gpu,
        self.sampler.sampling_states.seeds.gpu,
        pos,
        vocab_size,
        self.sampler.use_fp64_gumbel,
        apply_target_temperature,
    )
    return processed_logits, sampled, num_sampled


def install_rejection_sampler() -> None:
    global _INSTALLED
    if _INSTALLED or current_platform.device_name != "mcpu":
        return
    verify_upstream_compatibility(("rejection_sampler",))
    rebind(RejectionSampler, "_verify", _xcpu_rejection_sampler_verify)
    _INSTALLED = True
