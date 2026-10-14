# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch


def _xcpu_prepare_dflash_inputs(
    input_buffers: Any,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: Any,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    import torch_xcpu
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch_xcpu.ops.prepare_dflash_inputs(
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        sample_from_anchor=sample_from_anchor,
        pad_slot_id=PAD_SLOT_ID,
        cp_size=cp_size,
        cp_interleave=cp_interleave,
        num_reqs=input_batch.num_reqs,
    )


def maybe_patch_dflash_inputs() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator

    speculator_any = cast(Any, speculator)
    if getattr(speculator_any, "_xcpu_dflash_inputs_patched", False):
        return

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("dflash",))
    speculator_any.prepare_dflash_inputs = _xcpu_prepare_dflash_inputs
    speculator_any._xcpu_dflash_inputs_patched = True
