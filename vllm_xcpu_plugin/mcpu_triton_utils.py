# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, cast

import torch


class _FuncWrapper:
    def __init__(self, func: Callable, pass_grid: bool = False) -> None:
        self.func = func
        self.pass_grid = pass_grid

    def __getitem__(self, *args, **kwargs) -> Callable:
        if self.pass_grid:
            grid = args[0]

            def wrapped(*func_args, **func_kwargs):
                return self.func(*func_args, __grid=grid, **func_kwargs)

            return wrapped
        return self.func


def _compute_slot_mapping_kernel_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    TOTAL_CP_WORLD_SIZE: int,
    TOTAL_CP_RANK: int,
    CP_KV_CACHE_INTERLEAVE_SIZE: int,
    PAD_ID: int,
    BLOCK_SIZE: int,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on MCPU."
    torch.ops.mcpu.compute_slot_mapping_kernel_impl(
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        block_size,
    )


compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)


def _zero_kv_blocks_kernel_impl(
    seg_addrs: torch.Tensor,
    block_ids: torch.Tensor,
    n_blocks: int,
    N_SEGS: int,
    PAGE_SIZE_EL: int,
    BLOCK_SIZE: int,
) -> None:
    torch.ops.mcpu.zero_kv_blocks_kernel_impl(
        seg_addrs,
        block_ids,
        n_blocks,
        N_SEGS,
        PAGE_SIZE_EL,
    )


zero_kv_blocks_kernel = _FuncWrapper(_zero_kv_blocks_kernel_impl)


def _prepare_rope_positions_kernel_impl(
    positions: torch.Tensor,
    positions_stride: int,
    prefill_positions: torch.Tensor,
    prefill_positions_stride0: int,
    prefill_positions_stride1: int,
    prefill_delta: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    BLOCK_SIZE: int,
    NUM_DIMS: int,
) -> None:
    torch.ops.mcpu.prepare_rope_positions_kernel_impl(
        positions,
        positions_stride,
        prefill_positions,
        prefill_positions_stride0,
        prefill_positions_stride1,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
        NUM_DIMS,
    )


prepare_rope_positions_kernel = _FuncWrapper(_prepare_rope_positions_kernel_impl)


def _prepare_eagle_inputs_kernel_impl(
    last_token_indices: torch.Tensor,
    eagle_input_ids: torch.Tensor,
    eagle_positions: torch.Tensor,
    target_input_ids: torch.Tensor,
    target_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    BLOCK_SIZE: int,
    __grid: tuple[int, ...],
) -> None:
    num_reqs = __grid[0]
    torch.ops.mcpu.prepare_eagle_inputs_kernel_impl(
        last_token_indices,
        eagle_input_ids,
        eagle_positions,
        target_input_ids,
        target_positions,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        num_reqs,
    )


prepare_eagle_inputs_kernel = _FuncWrapper(
    _prepare_eagle_inputs_kernel_impl, pass_grid=True
)


def _prepare_eagle_docode_kernel_impl(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    output_hidden_states_stride: int,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    input_hidden_states: torch.Tensor,
    input_hidden_states_stride: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    hidden_size: int,
    max_model_len: int,
    max_num_reqs: int,
    BLOCK_SIZE: int,
    __grid: tuple[int, ...],
) -> None:
    num_reqs = __grid[0] - 1
    torch.ops.mcpu.prepare_eagle_decode_kernel_impl(
        draft_tokens,
        output_hidden_states,
        output_hidden_states_stride,
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_ids,
        positions,
        input_hidden_states,
        input_hidden_states_stride,
        query_start_loc,
        seq_lens,
        hidden_size,
        max_model_len,
        max_num_reqs,
        num_reqs,
    )


prepare_eagle_docode_kernel = _FuncWrapper(
    _prepare_eagle_docode_kernel_impl, pass_grid=True
)


def _update_eagle_inputs_kernel_impl(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    input_hidden_states: torch.Tensor,
    input_hidden_states_stride: int,
    seq_lens: torch.Tensor,
    max_model_len: int,
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    output_hidden_states_stride: int,
    hidden_size: int,
    BLOCK_SIZE: int,
) -> None:
    torch.ops.mcpu.update_eagle_inputs_kernel_impl(
        input_ids,
        positions,
        input_hidden_states,
        input_hidden_states_stride,
        seq_lens,
        max_model_len,
        draft_tokens,
        output_hidden_states,
        output_hidden_states_stride,
        hidden_size,
    )


update_eagle_inputs_kernel = _FuncWrapper(_update_eagle_inputs_kernel_impl)


def _strict_rejection_sample_kernel_impl(
    sampled: torch.Tensor,
    sampled_stride: int,
    num_sampled: torch.Tensor,
    target_sampled: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_warps: int,
) -> None:
    torch.ops.mcpu.strict_rejection_sample_kernel_impl(
        sampled,
        sampled_stride,
        num_sampled,
        target_sampled,
        draft_sampled,
        cu_num_logits,
    )


strict_rejection_sample_kernel = _FuncWrapper(_strict_rejection_sample_kernel_impl)


def patch_vllm_triton_kernels() -> None:
    import vllm.v1.worker.block_table
    import vllm.v1.worker.gpu.mm.rope
    import vllm.v1.worker.gpu.spec_decode.eagle.speculator
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler
    import vllm.v1.worker.utils

    vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
        compute_slot_mapping_kernel
    )
    vllm.v1.worker.utils._zero_kv_blocks_kernel = zero_kv_blocks_kernel
    vllm.v1.worker.gpu.mm.rope._prepare_rope_positions_kernel = (
        prepare_rope_positions_kernel
    )
    eagle_speculator = cast(Any, vllm.v1.worker.gpu.spec_decode.eagle.speculator)
    eagle_speculator._prepare_eagle_inputs_kernel = prepare_eagle_inputs_kernel
    eagle_speculator._prepare_eagle_docode_kernel = prepare_eagle_docode_kernel
    eagle_speculator._update_eagle_inputs_kernel = update_eagle_inputs_kernel

    rejection_sampler = cast(Any, vllm.v1.worker.gpu.spec_decode.rejection_sampler)
    rejection_sampler._strict_rejection_sample_kernel = strict_rejection_sample_kernel
