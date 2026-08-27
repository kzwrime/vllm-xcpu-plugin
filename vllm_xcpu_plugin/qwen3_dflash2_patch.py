# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch


def _xcpu_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    del num_groups, group_size, taps
    import torch_xcpu

    output = torch.empty_like(hidden_states)
    torch_xcpu.ops.dflash2_grouped_conv(hidden_states, delta, base, output, block_size)
    return output


def _xcpu_score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    import torch_xcpu

    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessor_features = predecessor_table[predecessor_ids] * hidden[:, :, None]
    edge_scores = torch.empty(
        *predecessor_features.shape[:-1],
        successors.shape[2],
        dtype=predecessor_features.dtype,
        device=predecessor_features.device,
    )
    torch_xcpu.ops.einsum_blpr_blcr_to_blpc(
        predecessor_features, successors, edge_scores
    )
    return unary_logits[:, :, None] + edge_scores


def maybe_patch_qwen3_dflash2() -> None:
    from vllm.model_executor.models import qwen3_dflash2

    module = cast(Any, qwen3_dflash2)
    if getattr(module, "_xcpu_dflash2_model_patched", False):
        return

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("qwen3_dflash2",))

    original_grouped_conv = module._grouped_conv
    original_score_edges = module._score_edges

    def _patched_grouped_conv(
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        block_size: int,
        num_groups: int,
        group_size: int,
        taps: int,
    ) -> torch.Tensor:
        if hidden_states.device.type in ("mcpu", "privateuseone"):
            return _xcpu_grouped_conv(
                hidden_states,
                delta,
                base,
                block_size,
                num_groups,
                group_size,
                taps,
            )
        return original_grouped_conv(
            hidden_states,
            delta,
            base,
            block_size,
            num_groups,
            group_size,
            taps,
        )

    def _patched_score_edges(
        predecessor_table: torch.Tensor,
        successor_table: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        if hidden.device.type in ("mcpu", "privateuseone"):
            return _xcpu_score_edges(
                predecessor_table,
                successor_table,
                candidate_ids,
                unary_logits,
                hidden,
                anchor_token_ids,
                top_k,
            )
        return original_score_edges(
            predecessor_table,
            successor_table,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            top_k,
        )

    module._grouped_conv = _patched_grouped_conv
    module._score_edges = _patched_score_edges
    module._xcpu_dflash2_model_patched = True
