# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any, cast

import torch

_TOPK_SOFTMAX_PATCHED = False
_TOPK_TOPP_SAMPLER_PATCHED = False


def _xcpu_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> None:
    if e_score_correction_bias is not None:
        raise NotImplementedError(
            "xcpu topk_softmax patch does not support e_score_correction_bias"
        )

    import torch_xcpu

    torch_xcpu.ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=is_padding,
    )


def maybe_patch_vllm_topk_softmax() -> None:
    global _TOPK_SOFTMAX_PATCHED

    if not bool(int(os.getenv("VLLM_USE_XCPU_TOPK_SOFTMAX", "0"))):
        return

    import vllm._custom_ops as ops

    if _TOPK_SOFTMAX_PATCHED:
        return

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("topk_softmax",))

    original_topk_softmax = ops.topk_softmax

    def _patched_topk_softmax(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
        e_score_correction_bias: torch.Tensor | None = None,
        is_padding: torch.Tensor | None = None,
    ) -> None:
        device_type = gating_output.device.type
        if device_type in ("mcpu", "privateuseone"):
            _xcpu_topk_softmax(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
                is_padding,
            )
            return

        original_topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            is_padding,
        )

    ops.topk_softmax = _patched_topk_softmax
    _TOPK_SOFTMAX_PATCHED = True


def maybe_patch_vllm_topk_topp_sampler() -> None:
    global _TOPK_TOPP_SAMPLER_PATCHED

    if not bool(int(os.getenv("VLLM_USE_XCPU_TOPK_TOPP_SAMPLER", "0"))):
        return

    if _TOPK_TOPP_SAMPLER_PATCHED:
        return

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("topk_topp",))

    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

    original_apply_top_k_top_p = topk_topp_sampler.apply_top_k_top_p

    def _patched_apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        device_type = logits.device.type
        if device_type in ("mcpu", "privateuseone"):
            if p is None and k is None:
                return logits
            import torch_xcpu

            return torch_xcpu.ops.apply_top_k_top_p(
                logits,
                k,
                p,
                allow_cpu_sync=True,
            )

        return original_apply_top_k_top_p(logits, k, p)

    topk_topp_sampler.apply_top_k_top_p = _patched_apply_top_k_top_p
    for module_name in (
        "vllm.v1.sample.rejection_sampler",
        "vllm.v1.worker.gpu.sample.sampler",
        "vllm.v1.worker.gpu.sample.states",
    ):
        module = sys.modules.get(module_name)
        if (
            module is not None
            and getattr(module, "apply_top_k_top_p", None) is original_apply_top_k_top_p
        ):
            cast(Any, module).apply_top_k_top_p = _patched_apply_top_k_top_p
    _TOPK_TOPP_SAMPLER_PATCHED = True
