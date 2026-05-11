# SPDX-License-Identifier: Apache-2.0

import torch


def _xcpu_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "softmax":
        scoring_func_idx = 0
    elif scoring_func == "sigmoid":
        scoring_func_idx = 1
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = hidden_states.shape[0]

    topk_weights = torch.empty(
        (num_token, topk), dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        (num_token, topk), dtype=torch.int32, device=hidden_states.device
    )

    import torch_xcpu

    if e_score_correction_bias is None:
        e_score_correction_bias = torch.empty(0)
    torch_xcpu.ops.grouped_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        bias=e_score_correction_bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func_idx,
        routed_scaling_factor=routed_scaling_factor,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    return topk_weights, topk_ids


def maybe_patch_vllm_grouped_topk() -> None:
    from vllm.model_executor.layers.fused_moe.router import grouped_topk_router

    if getattr(grouped_topk_router, "_xcpu_grouped_topk_patched", False):
        return

    original_grouped_topk = grouped_topk_router.grouped_topk

    def _patched_grouped_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        e_score_correction_bias: torch.Tensor,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = gating_output.device.type
        if device_type in ("mcpu", "privateuseone"):
            return _xcpu_grouped_topk(
                hidden_states=hidden_states,
                gating_output=gating_output,
                topk=topk,
                renormalize=renormalize,
                e_score_correction_bias=e_score_correction_bias,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
            )

        original_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    grouped_topk_router.grouped_topk = _patched_grouped_topk
    grouped_topk_router._xcpu_grouped_topk_patched = True
