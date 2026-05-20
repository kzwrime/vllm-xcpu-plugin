# SPDX-License-Identifier: Apache-2.0

import os

import torch

_TOPK_SOFTMAX_PATCHED = False


def _xcpu_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
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
    )


def maybe_patch_vllm_topk_softmax() -> None:
    global _TOPK_SOFTMAX_PATCHED

    if not bool(int(os.getenv("VLLM_USE_XCPU_TOPK_SOFTMAX", "0"))):
        return

    import vllm._custom_ops as ops

    if _TOPK_SOFTMAX_PATCHED:
        return

    original_topk_softmax = ops.topk_softmax

    def _patched_topk_softmax(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
        e_score_correction_bias: torch.Tensor | None = None,
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
            )
            return

        original_topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
        )

    ops.topk_softmax = _patched_topk_softmax
    _TOPK_SOFTMAX_PATCHED = True
