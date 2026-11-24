# SPDX-License-Identifier: Apache-2.0
"""Fused XCPU rejection sampler patch for vLLM RejectionSampler._verify."""

from __future__ import annotations

import os

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
_REJECTION_FUSED_ENV = "VLLM_XCPU_REJECTION_SAMPLE_FUSED"
_CHECK_TOKEN_RANGE_ENV = "VLLM_XCPU_CHECK_TOKEN_RANGE"


def _parse_bool_env(name: str, default: str) -> bool:
    raw = os.environ.get(name, default).strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    raise ValueError(
        f"Unsupported {name}={raw!r}; "
        "expected one of: 1, 0, true, false, on, off, yes, no"
    )


def _rejection_sample_fused_enabled() -> bool:
    return _parse_bool_env(_REJECTION_FUSED_ENV, "1")


def _single_op_rejection_override_active() -> bool:
    """Respect single-op A0/B0 switches unless fused impl is explicitly pinned."""
    if os.environ.get("TORCH_XCPU_VLLM_REJECTION_SAMPLE_FUSED_IMPL", "").strip():
        return False
    return bool(
        os.environ.get(
            "TORCH_XCPU_VLLM_REJECTION_COMPUTE_LOCAL_LOGITS_STATS_IMPL", ""
        ).strip()
        or os.environ.get("TORCH_XCPU_VLLM_REJECTION_RESAMPLE_IMPL", "").strip()
    )


def _sync_device_for_diag(tensor: torch.Tensor) -> None:
    if tensor.device.type == "mcpu" and hasattr(torch, "mcpu"):
        torch.mcpu.synchronize()
    elif tensor.device.type == "privateuseone" and hasattr(torch, "accelerator"):
        torch.accelerator.synchronize()


def _check_draft_sampled_range(
    verify_draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    vocab_size: int,
) -> None:
    _sync_device_for_diag(verify_draft_sampled)
    cu_cpu = cu_num_logits.detach().cpu()
    active_logits = int(cu_cpu[-1].item()) if cu_cpu.numel() > 0 else 0
    if active_logits <= 0:
        return
    ds_cpu = verify_draft_sampled[:active_logits].detach().cpu()
    invalid = (ds_cpu < -1) | (ds_cpu >= vocab_size)
    if bool(invalid.any().item()):
        bad_idx = int(invalid.nonzero(as_tuple=False)[0].item())
        bad_val = int(ds_cpu[bad_idx].item())
        raise RuntimeError(
            f"{_CHECK_TOKEN_RANGE_ENV}=1 violation at verify_entry.draft_sampled: "
            f"idx={bad_idx} val={bad_val} outside [-1, {vocab_size})"
        )


def _check_sampled_outputs_range(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    vocab_size: int,
) -> None:
    _sync_device_for_diag(sampled)
    sampled_cpu = sampled.detach().cpu()
    ns_cpu = num_sampled.detach().cpu()
    num_reqs = sampled_cpu.shape[0]
    width = sampled_cpu.shape[1]
    for r in range(num_reqs):
        n = int(ns_cpu[r].item())
        if n < 0 or n > width:
            raise RuntimeError(
                f"{_CHECK_TOKEN_RANGE_ENV}=1 violation at verify_exit.num_sampled: "
                f"req={r} num_sampled={n} outside [0, {width}]"
            )
        if n == 0:
            continue
        row = sampled_cpu[r, :n]
        invalid = (row < 0) | (row >= vocab_size)
        if bool(invalid.any().item()):
            col = int(invalid.nonzero(as_tuple=False)[0].item())
            bad_val = int(row[col].item())
            raise RuntimeError(
                f"{_CHECK_TOKEN_RANGE_ENV}=1 violation at verify_exit.sampled: "
                f"req={r} col={col} val={bad_val} outside [0, {vocab_size}), "
                f"row={row.tolist()}"
            )


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
    fused_enabled = _rejection_sample_fused_enabled()
    check_token_range = _parse_bool_env(_CHECK_TOKEN_RANGE_ENV, "0")
    fused_impl_pinned = bool(
        os.environ.get("TORCH_XCPU_VLLM_REJECTION_SAMPLE_FUSED_IMPL", "").strip()
    )

    vocab_size = (
        min(logits.shape[1], draft_logits.shape[-1])
        if draft_logits is not None
        else logits.shape[1]
    )
    if check_token_range:
        _check_draft_sampled_range(verify_draft_sampled, cu_num_logits, vocab_size)

    fused_eligible = (
        fused_enabled
        and not _single_op_rejection_override_active()
        and (not check_token_range or fused_impl_pinned)
        and logits.device.type in ("mcpu", "privateuseone")
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
        if check_token_range:
            _check_sampled_outputs_range(sampled, num_sampled, vocab_size)
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
    if check_token_range:
        _check_sampled_outputs_range(sampled, num_sampled, vocab_size)
    return processed_logits, sampled, num_sampled


def install_rejection_sampler() -> None:
    global _INSTALLED
    if _INSTALLED or current_platform.device_name != "mcpu":
        return
    verify_upstream_compatibility(("rejection_sampler",))
    rebind(RejectionSampler, "_verify", _xcpu_rejection_sampler_verify)
    _INSTALLED = True
