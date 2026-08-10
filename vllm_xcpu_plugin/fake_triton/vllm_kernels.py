# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch_mcpu  # noqa: F401

from .runtime import InvalidLaunchError, KernelLaunch, get_registry

# Informational only: this is the baseline at which the manifest was created.
# Never use it as a registration default or bulk-update it during a port. Each
# kernel below owns its literal source_version and may advance independently.
_MANIFEST_BASELINE_VERSION = "v0.24.0"

logger = logging.getLogger(__name__)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidLaunchError(message)


def _expect_grid(launch: KernelLaunch, expected: tuple[int, ...]) -> None:
    _expect(
        launch.grid == expected,
        f"{launch.kernel.qualname}: expected grid {expected}, got {launch.grid}",
    )


def _per_token_group_quant_fp8(launch: KernelLaunch) -> None:
    args = launch.arguments
    group_size = args["group_size"]
    num_columns = args["y_num_columns"]
    groups_per_row = num_columns // group_size
    num_groups = launch.grid[0]
    _expect_grid(launch, (num_groups,))
    _expect(num_columns % group_size == 0, "group_size must divide columns")
    _expect(num_groups % groups_per_row == 0, "grid does not cover whole rows")

    rows = num_groups // groups_per_row
    source = args["y_ptr"].as_strided(
        (rows, num_columns),
        (args["y_row_stride"], 1),
    )
    import torch_xcpu.ops as xcpu_ops

    xcpu_ops.per_token_group_quant_fp8_out(
        args["y_q_ptr"].reshape(rows, num_columns),
        args["y_s_ptr"].reshape(rows, groups_per_row),
        source,
        group_size,
        args["eps"],
        args["fp8_min"],
        args["fp8_max"],
        args["use_ue8m0"],
    )


def _zero_kv_blocks(launch: KernelLaunch) -> None:
    args = launch.arguments
    n_blocks = args["n_blocks"]
    n_segs = args["N_SEGS"]
    page_size = args["PAGE_SIZE_EL"]
    block_size = args["BLOCK_SIZE"]
    _expect(page_size % block_size == 0, "PAGE_SIZE_EL must divide BLOCK_SIZE")
    _expect_grid(
        launch,
        (n_blocks * n_segs * (page_size // block_size),),
    )
    torch.ops.mcpu.zero_kv_blocks_kernel_impl(
        args["seg_addrs_ptr"],
        args["block_ids_ptr"],
        n_blocks,
        n_segs,
        page_size,
    )


def _apply_write(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_writes = args["write_indices_ptr"].numel()
    _expect_grid(launch, (num_writes,))
    _expect(args["BLOCK_SIZE"] == 1024, "_apply_write_kernel BLOCK_SIZE must be 1024")
    if args["MULTI_GROUP"]:
        _expect(
            args["write_group_ids_ptr"] is not None,
            "multi-group apply_write requires group IDs",
        )
        torch.ops.mcpu.vllm_apply_write_multi(
            args["output_ptr"],
            args["output_stride"],
            args["write_indices_ptr"],
            args["write_starts_ptr"],
            args["write_contents_ptr"],
            args["write_cu_lens_ptr"],
            args["write_group_ids_ptr"],
        )
    else:
        _expect(
            args["write_group_ids_ptr"] is None,
            "single-group apply_write group IDs must be None",
        )
        torch.ops.mcpu.vllm_apply_write_single(
            args["output_ptr"],
            args["output_stride"],
            args["write_indices_ptr"],
            args["write_starts_ptr"],
            args["write_contents_ptr"],
            args["write_cu_lens_ptr"],
        )


def _gather_block_tables(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_groups, num_rows = launch.grid
    num_reqs = args["num_reqs"]
    _expect(num_reqs == args["batch_idx_to_req_idx"].numel(), "num_reqs mismatch")
    _expect(
        num_groups == args["src_block_table_ptrs"].numel(),
        "gather group grid mismatch",
    )
    _expect(args["BLOCK_SIZE"] == 1024, "gather BLOCK_SIZE must be 1024")
    torch.ops.mcpu.vllm_gather_block_tables_kernel(
        args["batch_idx_to_req_idx"],
        args["src_block_table_ptrs"],
        args["dst_block_table_ptrs"],
        args["block_table_strides"],
        args["num_blocks_ptr"],
        args["num_blocks_stride"],
        num_reqs,
        num_groups,
        num_rows,
    )


def _compute_slot_mappings(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_groups = args["block_sizes"].numel()
    num_reqs = args["idx_mapping"].numel()
    _expect_grid(launch, (num_groups, num_reqs + 1))
    _expect(
        args["TRITON_BLOCK_SIZE"] == 1024,
        "slot mapping TRITON_BLOCK_SIZE must be 1024",
    )
    torch.ops.mcpu.vllm_compute_slot_mappings_kernel(
        args["max_num_tokens"],
        args["idx_mapping"],
        args["query_start_loc"],
        args["pos"],
        args["block_table_ptrs"],
        args["block_table_strides"],
        args["block_sizes"],
        args["slot_mappings_ptr"],
        args["slot_mappings_stride"],
        args["cp_rank"],
        args["CP_SIZE"],
        args["CP_INTERLEAVE"],
        args["PAD_ID"],
        num_groups,
        num_reqs,
    )


def _compute_slot_mapping(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["query_start_loc_ptr"].numel() - 1
    _expect_grid(launch, (num_reqs + 1,))
    _expect(args["BLOCK_SIZE"] == 1024, "slot mapping BLOCK_SIZE must be 1024")
    _expect(
        args["KV_CACHE_BLOCK_SIZE"] == args["block_size"],
        "XCPU slot mapping requires matching KV-cache and kernel block sizes",
    )
    _expect(
        args["BLOCKS_PER_KV_BLOCK"] == 1,
        "XCPU slot mapping requires one kernel block per KV-cache block",
    )
    torch.ops.mcpu.vllm_compute_slot_mapping_kernel(
        args["num_tokens"],
        args["max_num_tokens"],
        args["query_start_loc_ptr"],
        args["positions_ptr"],
        args["block_table_ptr"],
        args["block_table_stride"],
        args["block_size"],
        args["slot_mapping_ptr"],
        args["TOTAL_CP_WORLD_SIZE"],
        args["TOTAL_CP_RANK"],
        args["CP_KV_CACHE_INTERLEAVE_SIZE"],
        args["PAD_ID"],
    )


def _combine_sampled_and_draft(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
        args["input_ids_ptr"],
        args["idx_mapping_ptr"],
        args["last_sampled_tokens_ptr"],
        args["query_start_loc_ptr"],
        args["seq_lens_ptr"],
        args["prefill_len_ptr"],
        args["draft_tokens_ptr"],
        args["draft_tokens_stride"],
        args["cu_num_logits_ptr"],
        args["logits_indices_ptr"],
        args["NUM_NEW_SAMPLED_TOKENS"],
    )


def _post_update(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    _expect(launch.metadata.get("num_warps") == 1, "post_update requires num_warps=1")
    torch.ops.mcpu.vllm_post_update(
        args["idx_mapping_ptr"],
        args["num_computed_tokens_ptr"],
        args["last_sampled_tokens_ptr"],
        args["output_bin_counts_ptr"],
        args["output_bin_counts_stride"],
        args["sampled_tokens_ptr"],
        args["sampled_tokens_stride"],
        args["num_sampled_ptr"],
        args["num_rejected_ptr"],
        args["query_start_loc_ptr"],
        args["all_token_ids_ptr"],
        args["all_token_ids_stride"],
        args["total_len_ptr"],
    )


def _grammar_bitmask(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_masks = args["logits_indices_ptr"].numel()
    vocab_size = args["vocab_size"]
    block_size = args["BLOCK_SIZE"]
    _expect_grid(launch, (num_masks, (vocab_size + block_size - 1) // block_size))
    torch.ops.mcpu.vllm_apply_grammar_bitmask(
        args["logits_ptr"],
        args["logits_indices_ptr"],
        args["bitmask_ptr"],
        vocab_size,
    )


def _logit_bias(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["expanded_idx_mapping_ptr"].numel(),))
    torch.ops.mcpu.vllm_bias_kernel(
        args["logits_ptr"],
        args["vocab_size"],
        args["expanded_idx_mapping_ptr"],
        args["num_allowed_token_ids_ptr"],
        args["allowed_token_ids_ptr"],
        args["num_logit_bias_ptr"],
        args["bias_token_ids_ptr"],
        args["bias_ptr"],
        args["pos_ptr"],
        args["min_lens_ptr"],
        args["num_stop_token_ids_ptr"],
        args["stop_token_ids_ptr"],
    )


def _fill_logprob_token_ids(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["sampled_token_ids_ptr"].numel(),))
    torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
        args["out_token_ids_ptr"],
        args["out_token_ids_stride"],
        args["out_valid_mask_ptr"],
        args["out_valid_mask_stride"],
        args["sampled_token_ids_ptr"],
        args["topk_indices_ptr"],
        args["topk_indices_stride"],
        args["expanded_idx_mapping_ptr"],
        args["num_per_req_token_ids_ptr"],
        args["per_req_token_ids_ptr"],
        args["per_req_token_ids_stride"],
        args["NUM_TOPK"],
        args["PADDED_COLS"],
    )


def _prepare_prefill_inputs(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    _expect(args["BLOCK_SIZE"] == 1024, "prefill BLOCK_SIZE must be 1024")
    _expect(
        args["all_token_ids_stride"] == args["all_token_ids_ptr"].stride(0),
        "all_token_ids stride mismatch",
    )
    torch.ops.mcpu.vllm_prepare_prefill_inputs(
        args["input_ids_ptr"],
        args["next_prefill_tokens_ptr"],
        args["idx_mapping_ptr"],
        args["query_start_loc_ptr"],
        args["all_token_ids_ptr"],
        args["prefill_lens_ptr"],
        args["num_computed_tokens_ptr"],
    )


def _autoregressive_prepare_prefill_inputs(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["idx_mapping_ptr"].numel()
    max_num_reqs = args["max_num_reqs"]
    _expect_grid(launch, (num_reqs,))
    _expect(
        args["BLOCK_SIZE"] == 1024,
        "autoregressive prefill BLOCK_SIZE must be 1024",
    )
    _expect(num_reqs > 0, "autoregressive prefill requires requests")
    _expect(max_num_reqs >= num_reqs, "max_num_reqs is smaller than active batch")
    _expect(
        args["query_start_loc_ptr"].numel() >= num_reqs + 1,
        "autoregressive prefill query_start_loc is too short",
    )
    _expect(
        args["draft_query_start_loc_ptr"].numel() >= max_num_reqs + 1,
        "autoregressive draft query_start_loc is too short",
    )
    torch.ops.mcpu.vllm_autoregressive_prepare_prefill_inputs(
        args["last_token_indices_ptr"],
        args["draft_current_step_ptr"],
        args["draft_input_ids_ptr"],
        args["draft_positions_ptr"],
        args["draft_query_start_loc_ptr"],
        args["draft_seq_lens_ptr"],
        args["target_input_ids_ptr"],
        args["target_positions_ptr"],
        args["idx_mapping_ptr"],
        args["last_sampled_ptr"],
        args["next_prefill_tokens_ptr"],
        args["num_sampled_ptr"],
        args["num_rejected_ptr"],
        args["query_start_loc_ptr"],
        args["seq_lens_ptr"],
        max_num_reqs,
    )


def _autoregressive_prepare_decode_inputs(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["draft_tokens_ptr"].shape[0]
    _expect_grid(launch, (num_reqs + 1,))
    _expect(
        args["BLOCK_SIZE"] == 1024,
        "autoregressive decode BLOCK_SIZE must be 1024",
    )
    _expect(
        args["draft_tokens_stride"] == args["draft_tokens_ptr"].stride(0),
        "autoregressive draft token stride mismatch",
    )
    _expect(
        args["max_num_reqs"] >= num_reqs,
        "max_num_reqs is smaller than active decode batch",
    )
    torch.ops.mcpu.vllm_autoregressive_prepare_decode_inputs(
        args["draft_tokens_ptr"],
        args["draft_tokens_stride"],
        args["target_seq_lens_ptr"],
        args["num_rejected_ptr"],
        args["input_ids_ptr"],
        args["positions_ptr"],
        args["query_start_loc_ptr"],
        args["seq_lens_ptr"],
        args["max_model_len"],
        args["max_num_reqs"],
        args["ADVANCE_DRAFT_POSITIONS"],
    )


def _autoregressive_update_draft_inputs(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["draft_tokens_ptr"].numel()
    _expect_grid(launch, (num_reqs,))
    _expect(
        args["BLOCK_SIZE"] == 1024,
        "autoregressive update BLOCK_SIZE must be 1024",
    )
    _expect(
        args["output_draft_tokens_stride"] == args["output_draft_tokens_ptr"].stride(0),
        "output draft token stride mismatch",
    )
    _expect(
        args["next_input_hidden_states_stride"]
        == args["next_input_hidden_states_ptr"].stride(0),
        "next input hidden state stride mismatch",
    )
    _expect(
        args["hidden_states_stride"] == args["hidden_states_ptr"].stride(0),
        "hidden state stride mismatch",
    )
    _expect(
        args["hidden_size"] == args["hidden_states_ptr"].shape[1],
        "hidden_size mismatch",
    )
    torch.ops.mcpu.vllm_autoregressive_update_draft_inputs(
        args["output_draft_tokens_ptr"],
        args["output_draft_tokens_stride"],
        args["next_input_hidden_states_ptr"],
        args["next_input_hidden_states_stride"],
        args["input_ids_ptr"],
        args["positions_ptr"],
        args["seq_lens_ptr"],
        args["draft_tokens_ptr"],
        args["current_draft_step_ptr"],
        args["hidden_states_ptr"],
        args["hidden_states_stride"],
        args["hidden_size"],
        args["max_model_len"],
        args["num_speculative_steps"],
        args["ADVANCE_DRAFT_POSITIONS"],
    )


def _prepare_pos_seq_lens(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel() + 1,))
    _expect(args["BLOCK_SIZE"] == 1024, "position BLOCK_SIZE must be 1024")
    _expect(args["max_num_reqs"] == args["seq_lens_ptr"].numel(), "max req mismatch")
    torch.ops.mcpu.vllm_prepare_pos_seq_lens(
        args["idx_mapping_ptr"],
        args["query_start_loc_ptr"],
        args["num_computed_tokens_ptr"],
        args["pos_ptr"],
        args["seq_lens_ptr"],
    )


def _prepare_rope_positions(launch: KernelLaunch) -> None:
    """Dispatch V2 M-RoPE/XD-RoPE position preparation to torch_mcpu.

    The Triton kernel writes only the query-token interval for every active
    request.  In particular, ``positions`` deliberately has a padded last
    column and is therefore non-contiguous in dimension zero; preserve its
    explicit row stride instead of requiring a contiguous tensor.
    """
    args = launch.arguments
    positions = args["positions_ptr"]
    prefill_positions = args["prefill_positions_ptr"]
    num_reqs = args["idx_mapping_ptr"].numel()
    num_dims = args["NUM_DIMS"]
    _expect_grid(launch, (num_reqs,))
    _expect(args["BLOCK_SIZE"] == 1024, "RoPE BLOCK_SIZE must be 1024")
    _expect(num_dims > 0, "RoPE NUM_DIMS must be positive")
    _expect(positions.ndim == 2, "RoPE positions must be 2D")
    _expect(prefill_positions.ndim == 2, "RoPE prefill positions must be 2D")
    _expect(positions.shape[0] == num_dims, "RoPE positions NUM_DIMS mismatch")
    _expect(
        prefill_positions.shape[0] >= num_dims,
        "RoPE prefill positions do not cover NUM_DIMS",
    )
    _expect(
        args["positions_stride"] == positions.stride(0),
        "RoPE positions stride mismatch",
    )
    # ``prefill_positions`` is [max_num_reqs * NUM_DIMS, max_model_len].
    # The Triton ABI groups its rows by request, so stride0 is NUM_DIMS rows
    # and stride1 is one row -- neither is the Tensor's dimension-1 stride.
    _expect(
        args["prefill_positions_stride1"] == prefill_positions.stride(0),
        "RoPE prefill positions stride1 mismatch",
    )
    _expect(
        args["prefill_positions_stride0"]
        == num_dims * args["prefill_positions_stride1"],
        "RoPE prefill positions stride0 must span NUM_DIMS rows",
    )
    _expect(
        prefill_positions.stride(1) == 1,
        "RoPE prefill positions columns must have unit stride",
    )
    _expect(
        args["query_start_loc_ptr"].numel() == num_reqs + 1,
        "RoPE query_start_loc size mismatch",
    )
    torch.ops.mcpu.prepare_rope_positions_kernel_impl(
        positions,
        args["positions_stride"],
        prefill_positions,
        args["prefill_positions_stride0"],
        args["prefill_positions_stride1"],
        args["prefill_delta_ptr"],
        args["idx_mapping_ptr"],
        args["query_start_loc_ptr"],
        args["prefill_lens_ptr"],
        args["num_computed_tokens_ptr"],
        num_dims,
    )


def _scatter_num_accepted(launch: KernelLaunch) -> None:
    args = launch.arguments
    idx_mapping = args["idx_mapping_ptr"]
    num_sampled = args["num_sampled_ptr"]
    num_accepted = args["num_accepted_ptr"]
    _expect_grid(launch, (idx_mapping.numel(),))
    _expect(idx_mapping.ndim == 1, "num-accepted idx_mapping must be 1D")
    _expect(num_sampled.ndim == 1, "num-accepted num_sampled must be 1D")
    _expect(num_accepted.ndim == 1, "num-accepted output must be 1D")
    _expect(
        num_sampled.numel() == idx_mapping.numel(),
        "num-accepted input size mismatch",
    )
    torch.ops.mcpu.vllm_scatter_num_accepted(
        idx_mapping,
        num_sampled,
        num_accepted,
    )


def _preprocess_mamba_align(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["num_reqs"]
    block_size = args["BLOCK_SIZE"]
    _expect(block_size == 256, "mamba align preprocess BLOCK_SIZE must be 256")
    _expect_grid(launch, ((num_reqs + block_size - 1) // block_size,))
    _expect(
        args["MAMBA_BLOCK_SIZE"] > 0,
        "mamba align preprocess MAMBA_BLOCK_SIZE must be positive",
    )
    torch.ops.mcpu.vllm_preprocess_mamba_align(
        args["idx_mapping_ptr"],
        args["state_idx_ptr"],
        args["num_computed_tokens_ptr"],
        args["query_start_loc_ptr"],
        args["num_accepted_tokens_ptr"],
        args["src_col_ptr"],
        args["src_off_ptr"],
        num_reqs,
        args["MAMBA_BLOCK_SIZE"],
    )


def _precopy_mamba_align(launch: KernelLaunch) -> None:
    args = launch.arguments
    total_states = args["state_base_addrs_ptr"].numel()
    _expect_grid(launch, (args["num_reqs"], total_states))
    _expect(
        args["COPY_BLOCK_SIZE"] == 1024,
        "mamba align precopy COPY_BLOCK_SIZE must be 1024",
    )
    torch.ops.mcpu.vllm_precopy_mamba_align(
        args["mamba_state_idx_ptr"],
        args["src_col_ptr"],
        args["token_bias_ptr"],
        args["block_table_ptrs_ptr"],
        args["block_table_stride_req"],
        args["state_base_addrs_ptr"],
        args["state_block_strides_ptr"],
        args["state_elem_sizes_ptr"],
        args["state_inner_sizes_ptr"],
        args["state_conv_widths_ptr"],
        args["state_group_indices_ptr"],
        args["state_dim_row_count_ptr"],
        args["state_dim_row_stride_ptr"],
        args["idx_mapping_ptr"],
        args["num_reqs"],
        total_states,
        args["CONV_STATE_DIM_FIRST"],
    )


def _postprocess_mamba(launch: KernelLaunch) -> None:
    args = launch.arguments
    total_states = args["state_base_addrs_ptr"].numel()
    _expect_grid(launch, (args["num_reqs"], total_states))
    _expect(
        args["COPY_BLOCK_SIZE"] == 1024,
        "mamba postprocess COPY_BLOCK_SIZE must be 1024",
    )
    _expect(args["block_size"] > 0, "mamba postprocess block_size must be positive")
    has_idx_mapping = args["HAS_IDX_MAPPING"]
    precomputed = args["PRECOMPUTED_NEW_COMPUTED"]
    v1_mode = (
        not has_idx_mapping
        and not precomputed
        and args["idx_mapping_ptr"] is None
        and args["num_scheduled_tokens_ptr"] is not None
        and args["num_draft_tokens_ptr"] is not None
        and args["num_accepted_tokens_out_ptr"] is not None
    )
    v2_align_mode = (
        has_idx_mapping
        and precomputed
        and args["idx_mapping_ptr"] is not None
        and args["num_scheduled_tokens_ptr"] is None
        and args["num_draft_tokens_ptr"] is None
        and args["num_accepted_tokens_out_ptr"] is None
    )
    _expect(
        v1_mode or v2_align_mode,
        "mamba postprocess only supports the audited V1 or V2-align ABI",
    )
    torch.ops.mcpu.vllm_postprocess_mamba(
        args["num_accepted_tokens_ptr"],
        args["mamba_state_idx_ptr"],
        args["num_scheduled_tokens_ptr"],
        args["num_computed_tokens_ptr"],
        args["num_draft_tokens_ptr"],
        args["block_table_ptrs_ptr"],
        args["block_table_stride_req"],
        args["state_base_addrs_ptr"],
        args["state_block_strides_ptr"],
        args["state_elem_sizes_ptr"],
        args["state_inner_sizes_ptr"],
        args["state_conv_widths_ptr"],
        args["state_group_indices_ptr"],
        args["state_dim_row_count_ptr"],
        args["state_dim_row_stride_ptr"],
        args["num_accepted_tokens_out_ptr"],
        args["idx_mapping_ptr"],
        args["num_reqs"],
        total_states,
        args["block_size"],
        args["CONV_STATE_DIM_FIRST"],
        has_idx_mapping,
        precomputed,
    )


def _get_num_sampled_and_rejected(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    torch.ops.mcpu.vllm_get_num_sampled_and_rejected(
        args["num_sampled_ptr"],
        args["num_rejected_ptr"],
        args["seq_lens_ptr"],
        args["cu_num_logits_ptr"],
        args["idx_mapping_ptr"],
        args["prefill_len_ptr"],
    )


def _post_update_num_computed_tokens(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    torch.ops.mcpu.vllm_post_update_pool(
        args["idx_mapping_ptr"],
        args["num_computed_tokens_ptr"],
        args["query_start_loc_ptr"],
    )


def _expand_idx_mapping(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    torch.ops.mcpu.vllm_expand_idx_mapping(
        args["idx_mapping_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["expanded_local_pos_ptr"],
        args["expanded_idx_mapping_ptr"].numel(),
        args["cu_num_logits_ptr"],
        args["BLOCK_SIZE"],
    )


def _rejection_sampler_expand(launch: KernelLaunch) -> None:
    args = launch.arguments
    input_tensor = args["input_ptr"]
    output = args["output_ptr"]
    cu_num_tokens = args["cu_num_tokens_ptr"]
    batch_size = input_tensor.numel()
    _expect_grid(launch, (batch_size,))
    _expect(output.ndim == 1, "rejection sampler expand output must be 1D")
    _expect(input_tensor.ndim == 1, "rejection sampler expand input must be 1D")
    _expect(cu_num_tokens.ndim == 1, "cu_num_tokens must be 1D")
    _expect(cu_num_tokens.numel() == batch_size, "cu_num_tokens size mismatch")
    _expect(output.dtype == input_tensor.dtype, "expand dtype mismatch")
    _expect(args["MAX_NUM_TOKENS"] > 0, "MAX_NUM_TOKENS must be positive")
    torch.ops.mcpu.vllm_rejection_sampler_expand(
        output,
        input_tensor,
        cu_num_tokens,
        args["replace_from"],
        args["replace_to"],
        args["MAX_NUM_TOKENS"],
    )


def _rejection_compute_local_logits_stats(launch: KernelLaunch) -> None:
    args = launch.arguments
    target_logits = args["target_logits_ptr"]
    vocab_size = args["vocab_size"]
    block_size = args["BLOCK_SIZE"]
    num_logits = target_logits.shape[0]
    num_blocks = (vocab_size + block_size - 1) // block_size
    _expect_grid(launch, (num_logits, num_blocks))
    _expect(block_size == 8192, "rejection block stats BLOCK_SIZE must be 8192")
    _expect(
        args["target_logits_stride"] == target_logits.stride(0),
        "target logits stride mismatch",
    )
    draft_logits = args["draft_logits_ptr"]
    if args["HAS_DRAFT_LOGITS"]:
        _expect(draft_logits is not None, "draft logits must be provided")
        _expect(
            args["draft_logits_stride_0"] == draft_logits.stride(0)
            and args["draft_logits_stride_1"] == draft_logits.stride(1),
            "draft logits stride mismatch",
        )
    else:
        _expect(draft_logits is None, "draft logits must be None")
        _expect(
            args["draft_logits_stride_0"] == 0
            and args["draft_logits_stride_1"] == 0,
            "absent draft logits must have zero strides",
        )
    for tensor_name, stride_name in (
        ("target_local_argmax_ptr", "target_local_argmax_stride"),
        ("target_local_max_ptr", "target_local_max_stride"),
        ("target_local_sumexp_ptr", "target_local_sumexp_stride"),
        ("draft_local_max_ptr", "draft_local_max_stride"),
        ("draft_local_sumexp_ptr", "draft_local_sumexp_stride"),
    ):
        _expect(
            args[stride_name] == args[tensor_name].stride(0),
            f"{tensor_name} stride mismatch",
        )
    torch.ops.mcpu.vllm_rejection_compute_local_logits_stats(
        args["target_local_argmax_ptr"],
        args["target_local_max_ptr"],
        args["target_local_sumexp_ptr"],
        args["draft_local_max_ptr"],
        args["draft_local_sumexp_ptr"],
        target_logits,
        draft_logits if args["HAS_DRAFT_LOGITS"] else None,
        args["expanded_idx_mapping_ptr"],
        args["expanded_local_pos_ptr"],
        args["temp_ptr"],
        vocab_size,
        args["num_speculative_steps"],
        block_size,
    )


def _rejection_cumulative_log_p(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["idx_mapping_ptr"].numel()
    _expect_grid(launch, (num_reqs,))
    _expect(
        launch.metadata.get("num_warps") == 1,
        "cumulative log-p requires num_warps=1",
    )
    torch.ops.mcpu.vllm_rejection_cumulative_log_p(
        args["cumulative_log_p_ptr"],
        args["target_logits_ptr"],
        args["target_local_max_ptr"],
        args["target_local_sumexp_ptr"],
        args["draft_sampled_ptr"],
        args["draft_logits_ptr"] if args["HAS_DRAFT_LOGITS"] else None,
        args["draft_local_max_ptr"],
        args["draft_local_sumexp_ptr"],
        args["cu_num_logits_ptr"],
        args["idx_mapping_ptr"],
        args["temp_ptr"],
        args["vocab_num_blocks"],
    )


def _rejection_local_residual_mass(launch: KernelLaunch) -> None:
    args = launch.arguments
    output = args["local_residual_mass_ptr"]
    _expect_grid(launch, output.shape)
    _expect(args["BLOCK_SIZE"] == 8192, "residual mass BLOCK_SIZE must be 8192")
    _expect(
        args["vocab_num_blocks"] == output.shape[1],
        "residual mass block count mismatch",
    )
    torch.ops.mcpu.vllm_rejection_local_residual_mass(
        output,
        args["cumulative_log_p_ptr"],
        args["target_logits_ptr"],
        args["target_local_max_ptr"],
        args["target_local_sumexp_ptr"],
        args["draft_logits_ptr"],
        args["draft_local_max_ptr"],
        args["draft_local_sumexp_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["expanded_local_pos_ptr"],
        args["temp_ptr"],
        args["vocab_size"],
        args["num_speculative_steps"],
    )


def _rejection_v2(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["idx_mapping_ptr"].numel()
    _expect_grid(launch, (num_reqs,))
    _expect(
        launch.metadata.get("num_warps") == 1,
        "rejection kernel requires num_warps=1",
    )
    _expect(not args["SYNTHETIC_MODE"], "synthetic rejection mode unsupported")
    if args["USE_BLOCK_VERIFICATION"]:
        _expect(args["cumulative_log_p_ptr"] is not None, "missing cumulative log p")
        if args["HAS_DRAFT_LOGITS"]:
            _expect(
                args["local_residual_mass_ptr"] is not None,
                "missing residual mass",
            )
    vocab_num_blocks = args["vocab_num_blocks"]
    _expect(vocab_num_blocks > 0, "vocab_num_blocks must be positive")
    _expect(
        args["PADDED_VOCAB_NUM_BLOCKS"] == 1 << (vocab_num_blocks - 1).bit_length(),
        "invalid PADDED_VOCAB_NUM_BLOCKS",
    )
    for tensor_name, stride_name in (
        ("sampled_ptr", "sampled_stride"),
        ("target_logits_ptr", "target_logits_stride"),
        ("target_local_argmax_ptr", "target_local_argmax_stride"),
        ("target_local_max_ptr", "target_local_max_stride"),
        ("target_local_sumexp_ptr", "target_local_sumexp_stride"),
        ("draft_local_max_ptr", "draft_local_max_stride"),
        ("draft_local_sumexp_ptr", "draft_local_sumexp_stride"),
    ):
        _expect(
            args[stride_name] == args[tensor_name].stride(0),
            f"{tensor_name} stride mismatch",
        )
    draft_logits = args["draft_logits_ptr"]
    if args["HAS_DRAFT_LOGITS"]:
        _expect(draft_logits is not None, "draft logits must be provided")
        _expect(
            args["draft_logits_stride_0"] == draft_logits.stride(0)
            and args["draft_logits_stride_1"] == draft_logits.stride(1),
            "draft logits stride mismatch",
        )
    else:
        _expect(draft_logits is None, "draft logits must be None")
        _expect(
            args["draft_logits_stride_0"] == 0
            and args["draft_logits_stride_1"] == 0,
            "absent draft logits must have zero strides",
        )
    torch.ops.mcpu.vllm_rejection(
        args["sampled_ptr"],
        args["rejected_steps_ptr"],
        args["target_rejected_logsumexp_ptr"],
        args["draft_rejected_logsumexp_ptr"],
        args["target_logits_ptr"],
        args["target_local_argmax_ptr"],
        args["target_local_max_ptr"],
        args["target_local_sumexp_ptr"],
        args["draft_sampled_ptr"],
        draft_logits if args["HAS_DRAFT_LOGITS"] else None,
        args["draft_local_max_ptr"],
        args["draft_local_sumexp_ptr"],
        args["cu_num_logits_ptr"],
        args["idx_mapping_ptr"],
        args["temp_ptr"],
        args["seed_ptr"],
        args["pos_ptr"],
        args["cumulative_log_p_ptr"],
        args["local_residual_mass_ptr"],
        args["USE_BLOCK_VERIFICATION"],
        vocab_num_blocks,
    )


def _rejection_resample(launch: KernelLaunch) -> None:
    args = launch.arguments
    vocab_size = args["vocab_size"]
    block_size = args["BLOCK_SIZE"]
    num_reqs = args["rejected_step_ptr"].numel()
    num_blocks = (vocab_size + block_size - 1) // block_size
    _expect_grid(launch, (num_reqs, num_blocks))
    _expect(block_size == 1024, "resample BLOCK_SIZE must be 1024")
    if args["USE_BLOCK_VERIFICATION"]:
        _expect(args["cumulative_log_p_ptr"] is not None, "missing cumulative log p")
    for tensor_name, stride_name in (
        ("resampled_local_argmax_ptr", "resampled_local_argmax_stride"),
        ("resampled_local_max_ptr", "resampled_local_max_stride"),
        ("target_logits_ptr", "target_logits_stride"),
    ):
        _expect(
            args[stride_name] == args[tensor_name].stride(0),
            f"{tensor_name} stride mismatch",
        )
    draft_logits = args["draft_logits_ptr"]
    if args["HAS_DRAFT_LOGITS"]:
        _expect(draft_logits is not None, "draft logits must be provided")
        _expect(
            args["draft_logits_stride_0"] == draft_logits.stride(0)
            and args["draft_logits_stride_1"] == draft_logits.stride(1),
            "draft logits stride mismatch",
        )
    else:
        _expect(draft_logits is None, "draft logits must be None")
        _expect(
            args["draft_logits_stride_0"] == 0
            and args["draft_logits_stride_1"] == 0,
            "absent draft logits must have zero strides",
        )
    torch.ops.mcpu.vllm_rejection_resample(
        args["resampled_local_argmax_ptr"],
        args["resampled_local_max_ptr"],
        args["target_logits_ptr"],
        args["target_rejected_logsumexp_ptr"],
        draft_logits if args["HAS_DRAFT_LOGITS"] else None,
        args["draft_rejected_logsumexp_ptr"],
        args["rejected_step_ptr"],
        args["cu_num_logits_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["draft_sampled_ptr"],
        args["temp_ptr"],
        args["seed_ptr"],
        args["pos_ptr"],
        args["cumulative_log_p_ptr"],
        vocab_size,
        block_size,
        args["USE_FP64"],
        args["USE_BLOCK_VERIFICATION"],
    )


def _rejection_insert(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["num_sampled_ptr"].numel()
    num_blocks = args["resample_num_blocks"]
    _expect_grid(launch, (num_reqs,))
    _expect(num_blocks > 0, "resample_num_blocks must be positive")
    _expect(
        args["PADDED_RESAMPLE_NUM_BLOCKS"] == 1 << (num_blocks - 1).bit_length(),
        "invalid PADDED_RESAMPLE_NUM_BLOCKS",
    )
    for tensor_name, stride_name in (
        ("sampled_ptr", "sampled_stride"),
        ("resampled_local_argmax_ptr", "resampled_local_argmax_stride"),
        ("resampled_local_max_ptr", "resampled_local_max_stride"),
    ):
        _expect(
            args[stride_name] == args[tensor_name].stride(0),
            f"{tensor_name} stride mismatch",
        )
    torch.ops.mcpu.vllm_rejection_insert(
        args["sampled_ptr"],
        args["num_sampled_ptr"],
        args["resampled_local_argmax_ptr"],
        args["resampled_local_max_ptr"],
        args["cu_num_logits_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["temp_ptr"],
        num_blocks,
    )


def _rejection_flatten(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["num_sampled_ptr"].numel()
    _expect_grid(launch, (num_reqs,))
    _expect(
        launch.metadata.get("num_warps") == 1,
        "flatten sampled requires num_warps=1",
    )
    _expect(
        args["sampled_stride"] == args["sampled_ptr"].stride(0),
        "flatten sampled stride mismatch",
    )
    torch.ops.mcpu.vllm_rejection_flatten(
        args["flat_sampled_ptr"],
        args["sampled_ptr"],
        args["num_sampled_ptr"],
        args["cu_num_logits_ptr"],
    )


def _rejection_greedy(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["cu_num_draft_tokens_ptr"].numel(),))
    _expect(not args["SYNTHETIC_MODE"], "synthetic rejection sampling unsupported")
    torch.ops.mcpu.vllm_rejection_greedy(
        args["output_token_ids_ptr"],
        args["cu_num_draft_tokens_ptr"],
        args["draft_token_ids_ptr"],
        args["target_argmax_ptr"],
        args["bonus_token_ids_ptr"],
        args["is_greedy_ptr"],
        args["max_spec_len"],
    )


def _sample_recovered(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect(
        len(launch.grid) == 2
        and launch.grid[0] == args["cu_num_draft_tokens_ptr"].numel()
        and launch.grid[1] > 0,
        "invalid recovered-token grid",
    )
    _expect(args["BLOCK_SIZE"] == 8192, "recovered BLOCK_SIZE must be 8192")
    torch.ops.mcpu.vllm_sample_recovered(
        args["output_token_ids_ptr"],
        args["cu_num_draft_tokens_ptr"],
        args["draft_token_ids_ptr"],
        args["draft_probs_ptr"],
        args["target_probs_ptr"],
        args["inv_q_ptr"],
        args["vocab_size"],
    )


def _rejection_random(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["cu_num_draft_tokens_ptr"].numel(),))
    _expect(not args["SYNTHETIC_MODE"], "synthetic rejection sampling unsupported")
    torch.ops.mcpu.vllm_rejection_random(
        args["output_token_ids_ptr"],
        args["cu_num_draft_tokens_ptr"],
        args["draft_token_ids_ptr"],
        args["draft_probs_ptr"],
        args["target_probs_ptr"],
        args["bonus_token_ids_ptr"],
        args["recovered_token_ids_ptr"],
        args["uniform_probs_ptr"],
        args["is_greedy_ptr"],
        args["max_spec_len"],
        args["vocab_size"],
    )


def _eagle_prepare_next_token_padded(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_reqs = args["num_reqs"]
    num_tokens = args["num_sampled_tokens_per_req"]
    _expect_grid(launch, (num_reqs,))
    _expect(num_tokens > 0, "num_sampled_tokens_per_req must be positive")
    _expect(
        args["stride_sampled_token_ids"] == args["sampled_token_ids_ptr"].stride(0),
        "sampled_token_ids stride mismatch",
    )
    _expect(
        args["BLOCK_SIZE_TOKENS"] == 1 << (num_tokens - 1).bit_length(),
        "BLOCK_SIZE_TOKENS must be the next power of two",
    )
    torch.ops.mcpu.vllm_eagle_prepare_next_token_padded(
        args["sampled_token_ids_ptr"],
        args["discard_request_mask_ptr"],
        args["backup_next_token_ids_ptr"],
        args["next_token_ids_ptr"],
        args["valid_sampled_tokens_count_ptr"],
        args["vocab_size"],
        num_tokens,
        num_reqs,
    )


def _eagle_prepare_inputs_padded(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["num_reqs"],))
    torch.ops.mcpu.vllm_eagle_prepare_inputs_padded(
        args["cu_num_draft_tokens_ptr"],
        args["valid_sampled_tokens_count_ptr"],
        args["query_start_loc_gpu_ptr"],
        args["token_indices_to_sample_ptr"],
        args["num_rejected_tokens_gpu_ptr"],
        args["num_reqs"],
    )


def _eagle_step_slot_mapping_metadata(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["out_slot_mapping_ptr"].numel(),))
    _expect(
        args["block_table_stride"] == args["block_table_ptr"].stride(0),
        "EAGLE block-table stride mismatch",
    )
    _expect(
        args["n_blocks_per_req"] == args["block_table_ptr"].shape[1],
        "EAGLE n_blocks_per_req mismatch",
    )
    torch.ops.mcpu.vllm_eagle_step_slot_mapping_metadata(
        args["positions_ptr"],
        args["block_table_ptr"],
        args["seq_lens_ptr"],
        args["out_clamped_positions_ptr"],
        args["out_slot_mapping_ptr"],
        args["block_size"],
        args["max_model_len"],
        args["n_blocks_per_req"],
        args["PAD_ID"],
        args["batch_size"],
    )


def _bad_words(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect(
        launch.grid[0] == args["expanded_idx_mapping_ptr"].numel(), "bad words grid"
    )
    _expect(
        args["bad_word_token_ids_stride"] == args["bad_word_token_ids_ptr"].stride(0),
        "bad-word token stride mismatch",
    )
    _expect(
        args["bad_word_offsets_stride"] == args["bad_word_offsets_ptr"].stride(0),
        "bad-word offset stride mismatch",
    )
    _expect(
        args["all_token_ids_stride"] == args["all_token_ids_ptr"].stride(0),
        "all-token stride mismatch",
    )
    torch.ops.mcpu.vllm_bad_words_kernel(
        args["logits_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["bad_word_token_ids_ptr"],
        args["bad_word_offsets_ptr"],
        args["num_bad_words_ptr"],
        args["all_token_ids_ptr"],
        args["prompt_len_ptr"],
        args["total_len_ptr"],
        args["input_ids_ptr"],
        args["expanded_local_pos_ptr"],
    )


def _topk_log_softmax(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["logits_ptr"].shape[0],))
    _expect(args["logits_stride"] == args["logits_ptr"].stride(0), "logits stride")
    torch.ops.mcpu.vllm_topk_log_softmax_kernel(
        args["output_ptr"],
        args["logits_ptr"],
        args["topk_ids_ptr"],
        args["topk"],
        args["vocab_size"],
    )


def _ranks(launch: KernelLaunch) -> None:
    args = launch.arguments
    logits = args["logits_ptr"]
    batch = logits.shape[0]
    _expect_grid(launch, (batch,))
    _expect(args["logits_stride"] == logits.stride(0), "logits stride")
    _expect(logits.stride(1) == 1, "logits columns must be contiguous")
    _expect(args["output_ptr"].numel() >= batch, "output is shorter than grid")
    _expect(args["token_ids_ptr"].numel() >= batch, "token ids are shorter than grid")
    _expect(args["output_ptr"].stride(0) == 1, "output must have unit stride")
    _expect(args["token_ids_ptr"].stride(0) == 1, "token ids must have unit stride")
    _expect(0 < args["vocab_size"] <= logits.shape[1], "invalid vocab size")
    _expect(args["BLOCK_SIZE"] > 0, "BLOCK_SIZE must be positive")
    torch.ops.mcpu.vllm_ranks_kernel(
        args["output_ptr"],
        args["logits_ptr"],
        args["token_ids_ptr"],
        args["vocab_size"],
    )


def _min_p(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["expanded_idx_mapping_ptr"].numel(),))
    _expect(args["logits_stride"] == args["logits_ptr"].stride(0), "logits stride")
    torch.ops.mcpu.vllm_min_p_kernel(
        args["logits_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["min_p_ptr"],
        args["vocab_size"],
    )


def _penalties(launch: KernelLaunch) -> None:
    args = launch.arguments
    block_size = args["BLOCK_SIZE"]
    _expect_grid(
        launch,
        (
            args["expanded_idx_mapping_ptr"].numel(),
            (args["vocab_size"] + block_size - 1) // block_size,
        ),
    )
    torch.ops.mcpu.vllm_penalties_kernel(
        args["logits_ptr"],
        args["expanded_idx_mapping_ptr"],
        args["token_ids_ptr"],
        args["expanded_local_pos_ptr"],
        args["repetition_penalty_ptr"],
        args["frequency_penalty_ptr"],
        args["presence_penalty_ptr"],
        args["prompt_bin_mask_ptr"],
        args["output_bin_counts_ptr"],
        args["vocab_size"],
    )


def _bincount(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect(launch.grid[0] == args["expanded_idx_mapping_ptr"].numel(), "bincount grid")
    _expect(args["BLOCK_SIZE"] == 1024, "bincount BLOCK_SIZE must be 1024")
    torch.ops.mcpu.vllm_bincount_kernel(
        args["expanded_idx_mapping_ptr"],
        args["all_token_ids_ptr"],
        args["prompt_len_ptr"],
        args["prefill_len_ptr"],
        args["prompt_bin_mask_ptr"],
        args["output_bin_counts_ptr"],
    )


def _prompt_logprob_token_ids(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect_grid(launch, (args["idx_mapping_ptr"].numel(),))
    _expect(args["BLOCK_SIZE"] == 1024, "prompt logprob BLOCK_SIZE must be 1024")
    _expect(
        args["all_token_ids_stride"] == args["all_token_ids_ptr"].stride(0),
        "all-token stride mismatch",
    )
    torch.ops.mcpu.vllm_prompt_logprobs_token_ids(
        args["prompt_logprobs_token_ids_ptr"],
        args["query_start_loc_ptr"],
        args["idx_mapping_ptr"],
        args["num_computed_tokens_ptr"],
        args["all_token_ids_ptr"],
    )


_KERNELS: tuple[
    tuple[str, str, str, str, str, Callable[[KernelLaunch], Any], tuple[str, ...]], ...
] = (
    (
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        "_per_token_group_quant_fp8",
        "7717aa7f963f67003bcd95773ee448a344f772c0789b60abf4765c7abe5afb3b",
        "ef21c8054336abd0321214df36f5ce0888919da816686b2a61e559a392b1d1fa",
        "v0.25.0",
        _per_token_group_quant_fp8,
        ("num_stages", "num_warps"),
    ),
    (
        "vllm.v1.worker.utils",
        "_zero_kv_blocks_kernel",
        "eedc25dc850294fdcb43a7c7e4fb04c465740a435157dd91e05a103fb2366304",
        "b572c3cc2feaa91e97397b5a07a4ac4fa9597667c8dad6eb3388d7a51fa5e351",
        "v0.24.0",
        _zero_kv_blocks,
        (),
    ),
    (
        "vllm.v1.worker.gpu.buffer_utils",
        "_apply_write_kernel",
        "6bde42298a330d5060b8061fdfcfba17cf7a0a812d4a78b54ff148a9b4dfd257",
        "1fe7155b89393baa74552e3015f08f47b3130da9051adb8de6ef8cec8b34fcf7",
        "v0.24.0",
        _apply_write,
        (),
    ),
    (
        "vllm.v1.worker.gpu.block_table",
        "_gather_block_tables_kernel",
        "5aa908b7646c32148e5fb7cdd62781e7fd466b359a674a4bf7416358a7d0f2a6",
        "cac827ebbf78044d46ae1fb2c2b38f96cb4ef0193ae9e0a6851d9042b5977e70",
        "v0.24.0",
        _gather_block_tables,
        (),
    ),
    (
        "vllm.v1.worker.block_table",
        "_compute_slot_mapping_kernel",
        "80be96b2bf56f10d060d32779de2bac5577db68b0cdd345b7a91437158858027",
        "5eb4c7a44f8ce37b48472fd97ca0bd19e4f87a7f5a3b22b3ab6ef39f53a699cd",
        "v0.25.0",
        _compute_slot_mapping,
        (),
    ),
    (
        "vllm.v1.worker.gpu.block_table",
        "_compute_slot_mappings_kernel",
        "e7cfdd055ee4f32c0fb5b18091f508dd8754d58f9120034105d425e44cdaff9b",
        "653c92181fba55279fb7a633ad85485888f04b418f8c19095ce5f0dc13785abf",
        "v0.24.0",
        _compute_slot_mappings,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_combine_sampled_and_draft_tokens_kernel",
        "87a15b332c59f4be05844b8b761960f1bbaa0a4182aa29215fec36c1aa871b22",
        "4ef7c5370798e1bd7a972572d12a64e4c4463d6ed04dab404ccb55867a3b1e9e",
        "v0.25.0",
        _combine_sampled_and_draft,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_post_update_kernel",
        "716d50004e13452fa8fefb211d3e5ca66756f7d78b9f63fda008431b3d7b0286",
        "3097fa96b4547ccdf952985b5aa933d9b73f651f919b2124025d337934273174",
        "v0.24.0",
        _post_update,
        ("num_warps",),
    ),
    (
        "vllm.v1.worker.gpu.structured_outputs",
        "_apply_grammar_bitmask_kernel",
        "e334b50d21b3e3f8c39e7e64ade9135b25e4039d460f6ca271e6b44335953b21",
        "fb602ffdcdbdcaf4fdb3dee73b58435e5106e107c2fd037e88b3c5cc8c75563b",
        "v0.24.0",
        _grammar_bitmask,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logit_bias",
        "_bias_kernel",
        "a37b8836ad5f45654eabba292a4abe7e1f37da9695e253aab79afd07530bf6bf",
        "9681ff4dfb8298a8407b5f8655ffdbb129f4474bea20bd0292ea789d4194fba9",
        "v0.24.0",
        _logit_bias,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logprob",
        "_fill_logprob_token_ids_kernel",
        "67b978589645faf3ad20c43fd78c6ba91d9a1ddc85623c07df8501a317c4e2bb",
        "9aa0a7b30e841c7e5137fcca3d96ad6ee5a2ef04a13166d0a940980e9fccbdf4",
        "v0.24.0",
        _fill_logprob_token_ids,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_prepare_prefill_inputs_kernel",
        "928ffeb93a37d7de91cea22760440a77a5af2d25f2bed4671ca0d41984a6ff35",
        "77c7c4a386f60887453fb464c275933bbcb05f79e68be455f5b0ce0012eda6ed",
        "v0.24.0",
        _prepare_prefill_inputs,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
        "_prepare_prefill_inputs_kernel",
        "fc094fbcf36c318670ece79a9cbbde4013966270c30bbba5c435c5778b8e57cc",
        "ed7370baed785629709f3a261987fe0118a40852c162e799bf5ebf6fb32ed68e",
        "v0.24.0",
        _autoregressive_prepare_prefill_inputs,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
        "_prepare_decode_inputs_kernel",
        "ac1dcb4ec4b02e6651b540fc1fce1b7a54c2b14f0ed2bfe40378a46a51a6f465",
        "37e3c10b0abdfb8ba5e74337038cfbc91ffae15c4607e21b4d20554414bbd7a5",
        "v0.24.0",
        _autoregressive_prepare_decode_inputs,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
        "_update_draft_inputs_kernel",
        "4a61e8fbc62acf6639ef943db78daa82232c4682e7cf319ea2036ace56965522",
        "fec4868f0636b79a8882919a16186c294ada21359a3b52655e4109cd6430beba",
        "v0.24.0",
        _autoregressive_update_draft_inputs,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_prepare_pos_seq_lens_kernel",
        "5e42c1095fe391474d4cac66c76dba29259a62ff2a20ce5cf0ce5df7d52accef",
        "41458ec9cbc4829bfd42e070efdc5be7d1c6fd0a884d674324a6d0496296a096",
        "v0.24.0",
        _prepare_pos_seq_lens,
        (),
    ),
    (
        "vllm.v1.worker.gpu.mm.rope",
        "_prepare_rope_positions_kernel",
        "65158c5c27896b717e28c9b6727aa95285969e1221526faa531c0cf5b766bd34",
        "7b68271fdc1121235ae9233ae2da4fa18745e1bb8c65e69373649a8545e4e81d",
        "v0.24.0",
        _prepare_rope_positions,
        (),
    ),
    (
        "vllm.v1.worker.mamba_utils",
        "preprocess_mamba_align_fused_kernel",
        "55c1440a10f71a2582d7c8aec21c1ccfa605381b5c3f24c2990c2be08ec10ab6",
        "d37373ee6a2ab978247bbba864c785901c1afd677b98850f9d6a68bb81e77e49",
        "v0.25.1",
        _preprocess_mamba_align,
        (),
    ),
    (
        "vllm.v1.worker.mamba_utils",
        "precopy_mamba_align_fused_kernel",
        "9bbbde89056d4fba815268a71d038650bc840e2dafd9944c3fd9b619876efd2e",
        "7e3b13b611b2c064d90bc22719d3ec5384dc9a93eba374f7885195bd73b1e35a",
        "v0.25.1",
        _precopy_mamba_align,
        (),
    ),
    (
        "vllm.v1.worker.mamba_utils",
        "postprocess_mamba_fused_kernel",
        "fbb8149128bd42bc175a2e3ae42b8600634a0d973ce7098b7a076c1bfa7acf6c",
        "34919a7a6f15f7639eb0ee59abebf561f2c48b1243eee651aa03adeafd7ec97b",
        "v0.25.1",
        _postprocess_mamba,
        (),
    ),
    (
        "vllm.v1.worker.gpu.model_states.mamba_hybrid",
        "_scatter_num_accepted_kernel",
        "20bca6b80f9d2a5d6a51a88bf683256fd9d15efe05a1592da25d7d2f31d5b107",
        "5619eaf2ab8f252bf3562bd92a9c78da16e96c5daada7a8c8aed720983daa19e",
        "v0.24.0",
        _scatter_num_accepted,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_get_num_sampled_and_rejected_kernel",
        "8e76f6adf25e74d7d7bdf053d50399861f50613fb707acdd0b869b1c3be82b07",
        "8172888d9239019811f577e2b0dbafcc7d656804f3503c753016f34d3cbafce1",
        "v0.24.0",
        _get_num_sampled_and_rejected,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_post_update_num_computed_tokens_kernel",
        "9c9754a3c48a493290d6917370a96d79620239809fc0edb08e6906cf45acfcad",
        "042f8815ef34ea47ce1c367ab3b183c34001b97d0b598907a1d5090227691a3a",
        "v0.24.0",
        _post_update_num_computed_tokens,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_expand_idx_mapping_kernel",
        "c716f4fadb06c80d28adbb03307d989bf62f7ef9ae6694549650fdc2a7983045",
        "a91a97bdb0a7d9f6ad5095f72509905e256570d914a116741ad0c93ee51d4128",
        "v0.24.0",
        _expand_idx_mapping,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        "_flatten_sampled_kernel",
        "0256a6b5405bc8f62f2cd2dff4c83f613d9c2c8e1c74ac0f9dc477341f576778",
        "bba626cce7c4642ff2c2c1f6839fe7010e03368c201ba370a4dab9ca713d16a7",
        "v0.24.0",
        _rejection_flatten,
        ("num_warps",),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        "_compute_local_logits_stats_kernel",
        "d37d345853cbd4d215456059727b94ec42575c1e8dd5421991b489b0ba5e23bc",
        "f7df3922cbf844c774e1d7746d88d1b93dd75a99b0232d955c69d91ef553aea8",
        "v0.25.1",
        _rejection_compute_local_logits_stats,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        "_compute_cumulative_log_p_kernel",
        "facbac093d0917b1ac05ed5946e5283983ac5df7c250a661ee92252c381f575a",
        "9288fe68370ddebf205d8583a3fde34969f8d3535c2f208615598df7140bddd2",
        "v0.25.1",
        _rejection_cumulative_log_p,
        ("num_warps",),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        "_compute_local_residual_mass_kernel",
        "3c0f47a77b2d498787c90b06659a4dfeb2565079a4e9957f1811757766c19ef4",
        "bedaed3ba6a0b2032fac12cc9f5b33b47511463fb5385f5f57470104565869d5",
        "v0.25.1",
        _rejection_local_residual_mass,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        "_rejection_kernel",
        "3920098c7bd8c39df49c5ddd3a29ac5410b34589380ab5f45d595f2a9fc4a998",
        "ab95ffecb83ee24783fcfa2f1a52274a82888fc2763fd0669421935647b0ae5b",
        "v0.25.1",
        _rejection_v2,
        ("num_warps",),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        "_resample_kernel",
        "1b1246d50521edc32128655c5fc289d424b2ebba5aa785a98c16d6fb8439da53",
        "2cb197288d2aabd1719d7dd1a599f48a5c059f1a8c4549196347eb95ea28af2a",
        "v0.25.1",
        _rejection_resample,
        (),
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        "_insert_resampled_kernel",
        "e4daecaefc15bbc4b02c6ee7541dc1acd0ad582afcd617efb325d0d084a2273b",
        "31af6f6b1e71365896c70a7da6263928ffc9a30877b62a608cd452b0f9681ac8",
        "v0.24.0",
        _rejection_insert,
        (),
    ),
    (
        "vllm.v1.sample.rejection_sampler",
        "expand_kernel",
        "6bc6d69deac4b0bf8aa07bcc879e80f1e3e8ab45ee003b90b56b5756c84e1910",
        "e915db20eca855654152e1b0db836f032483d3cb477487cbf305b2a5e21e4b00",
        "v0.24.0",
        _rejection_sampler_expand,
        (),
    ),
    (
        "vllm.v1.sample.rejection_sampler",
        "rejection_greedy_sample_kernel",
        "8f96dd95137ff29dd09be620021f72c79162376ba6b8e75077aae69ff1696d9f",
        "5cb68da8498b1d54a330aee377d9cdc4e7122bb34a74e53b0169903a927e3bc2",
        "v0.24.0",
        _rejection_greedy,
        (),
    ),
    (
        "vllm.v1.sample.rejection_sampler",
        "sample_recovered_tokens_kernel",
        "2b93b033709712e066b19d1dc1873a11468c36b2be448c5deec72745aa8f3755",
        "00fbf642aff31227f9dca2537c92deccaa0bed7a5c35bb94af548146d8e4d40a",
        "v0.24.0",
        _sample_recovered,
        (),
    ),
    (
        "vllm.v1.sample.rejection_sampler",
        "rejection_random_sample_kernel",
        "f75f9429eafb383840ff75eaa550d27491ea09cafb029244c8292e09f23e8a86",
        "61d2450a512a6a2211a01b17f242f2001b70f55ef270cfa75adc524da4253347",
        "v0.24.0",
        _rejection_random,
        (),
    ),
    (
        "vllm.v1.spec_decode.utils",
        "eagle_prepare_next_token_padded_kernel",
        "466d87c788c9dcf9a842cabe2777ecb85460a64bba6929902740cebd6313a37b",
        "3622acaef3e6e01bda16622d695a1913cce1353a08feddeff9592e0310d53cb8",
        "v0.24.0",
        _eagle_prepare_next_token_padded,
        (),
    ),
    (
        "vllm.v1.spec_decode.utils",
        "eagle_prepare_inputs_padded_kernel",
        "499c9ce7d993134d799f89d9f895d2070bc4aed5851ee4dfa033fbf5c850b4da",
        "0503ad7b03daa8bba43ea18b80e27d3b64ea733466ed612b079a5c8d0c13e088",
        "v0.24.0",
        _eagle_prepare_inputs_padded,
        (),
    ),
    (
        "vllm.v1.spec_decode.utils",
        "eagle_step_slot_mapping_metadata_kernel",
        "fecbccb0286df45eb67a092836a29357f75b3e445a249738dd95739728bb82d8",
        "972738fccc9dc1fe972bd1afc1e7dfe689fc453ab8c176e6e4d405a17e617add",
        "v0.24.0",
        _eagle_step_slot_mapping_metadata,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.bad_words",
        "_bad_words_kernel",
        "d58260c941d0deaaa6e474f18c6db2cc5b3bd959a63b9d5ee35450320ecdb55a",
        "e275dcc38c8769bebb9322fd473062d0c9319963f8b1ed0ee4c8f0e093388a54",
        "v0.24.0",
        _bad_words,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logprob",
        "_topk_log_softmax_kernel",
        "31c228669aa02373e285e4e8f318a4b62beda5bacbe9be3dc4e1f81715dfd908",
        "ed8ecd4bb2608522dcd02fd61f20aebb36fd6fc99eb3b6322392ea0082cf7c31",
        "v0.24.0",
        _topk_log_softmax,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logprob",
        "_ranks_kernel",
        "f1b2b5dc4deef9f30924304f7bf98393138b883dd98db60f65f139ffc26e7c90",
        "1b8570353f9fe64bae9cfd8ea5fc80c5b89aa72d711adfea0d464204ccb8e1a6",
        "v0.24.0",
        _ranks,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.min_p",
        "_min_p_kernel",
        "07cfd4634f986ac5cd6e0f5c7f66b678c7bb6b1168cae50fd699c23fee949f0a",
        "334bc27604309735887a01c28b4ffbf2283780b45cd95f50f98cd9095022cb79",
        "v0.24.0",
        _min_p,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.penalties",
        "_penalties_kernel",
        "7d427188e1874917ad43b0a843b718518ef9bc4116be8367c06fe72bfba5e3dc",
        "332996dd6a4f16f92ec084fe2623bc06f3b65c46f2a667ba841505284d27fb4f",
        "v0.24.0",
        _penalties,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.penalties",
        "_bincount_kernel",
        "3009661a6db97446184fb9ed6458069c580a12c4a9e2cc643dff35ca4a80369c",
        "14a55431028882a62010aa7273ff46305b8f4b361067479c30b7c885367eaf5f",
        "v0.24.0",
        _bincount,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.prompt_logprob",
        "_prompt_logprobs_token_ids_kernel",
        "85a3b2e004343807a21da4eb84d09548c3bb1f26d5a0db01cfb826cd7a9f3b54",
        "a711807c2084cbfea9cd3094ce73d6023f335fa65ed111c53317ab26276ca909",
        "v0.24.0",
        _prompt_logprob_token_ids,
        (),
    ),
)


def register_vllm_kernels() -> None:
    """Register available vLLM kernels and validate them when launched.

    vLLM imports every plugin during CLI construction, including kernels for
    optional features that the current server will never use. Deferring the
    fingerprint check keeps unrelated upstream drift from blocking startup;
    :meth:`KernelRegistry.dispatch` still fails closed before a mismatched
    replacement can execute.
    """
    registry = get_registry()
    for (
        module_name,
        name,
        source_hash,
        signature_hash,
        source_version,
        adapter,
        metadata,
    ) in _KERNELS:
        module = importlib.import_module(module_name)
        try:
            kernel = getattr(module, name)
        except AttributeError:
            logger.warning(
                "Skipping unavailable optional Fake Triton kernel %s.%s; "
                "a replacement will be required before that path can run",
                module_name,
                name,
            )
            continue
        registry.register(
            kernel,
            adapter,
            expected_source_hash=source_hash,
            expected_signature_hash=signature_hash,
            allowed_metadata=metadata,
            owner="torch_mcpu",
            source_version=source_version,
            defer_version_check=True,
        )
