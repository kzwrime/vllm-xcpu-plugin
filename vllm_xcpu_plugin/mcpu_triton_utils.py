# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch


class _FuncWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
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


def patch_vllm_triton_kernels() -> None:
    import vllm.v1.worker.block_table
    import vllm.v1.worker.gpu.mm.rope
    import vllm.v1.worker.utils

    vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
        compute_slot_mapping_kernel
    )
    vllm.v1.worker.utils._zero_kv_blocks_kernel = zero_kv_blocks_kernel
    vllm.v1.worker.gpu.mm.rope._prepare_rope_positions_kernel = (
        prepare_rope_positions_kernel
    )
