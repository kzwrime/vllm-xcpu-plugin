# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch_mcpu  # noqa: F401

from .runtime import InvalidLaunchError, KernelLaunch, get_registry

_VLLM_VERSION = "v0.24.0"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidLaunchError(message)


def _expect_grid(launch: KernelLaunch, expected: tuple[int, ...]) -> None:
    _expect(
        launch.grid == expected,
        f"{launch.kernel.qualname}: expected grid {expected}, got {launch.grid}",
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
    tuple[str, str, str, str, Callable[[KernelLaunch], Any], tuple[str, ...]], ...
] = (
    (
        "vllm.v1.worker.utils",
        "_zero_kv_blocks_kernel",
        "eedc25dc850294fdcb43a7c7e4fb04c465740a435157dd91e05a103fb2366304",
        "b572c3cc2feaa91e97397b5a07a4ac4fa9597667c8dad6eb3388d7a51fa5e351",
        _zero_kv_blocks,
        (),
    ),
    (
        "vllm.v1.worker.gpu.buffer_utils",
        "_apply_write_kernel",
        "6bde42298a330d5060b8061fdfcfba17cf7a0a812d4a78b54ff148a9b4dfd257",
        "1fe7155b89393baa74552e3015f08f47b3130da9051adb8de6ef8cec8b34fcf7",
        _apply_write,
        (),
    ),
    (
        "vllm.v1.worker.gpu.block_table",
        "_gather_block_tables_kernel",
        "5aa908b7646c32148e5fb7cdd62781e7fd466b359a674a4bf7416358a7d0f2a6",
        "cac827ebbf78044d46ae1fb2c2b38f96cb4ef0193ae9e0a6851d9042b5977e70",
        _gather_block_tables,
        (),
    ),
    (
        "vllm.v1.worker.block_table",
        "_compute_slot_mapping_kernel",
        "393b36f4c4e3d151ca01aebae0ddd7aa1e9660efc68798af438f22e8abbe393a",
        "790656ce797bc0221c490834c23cb70ca4df7e6047514cfcfce27c8b7465edc5",
        _compute_slot_mapping,
        (),
    ),
    (
        "vllm.v1.worker.gpu.block_table",
        "_compute_slot_mappings_kernel",
        "e7cfdd055ee4f32c0fb5b18091f508dd8754d58f9120034105d425e44cdaff9b",
        "653c92181fba55279fb7a633ad85485888f04b418f8c19095ce5f0dc13785abf",
        _compute_slot_mappings,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_combine_sampled_and_draft_tokens_kernel",
        "2300cccbb1ef0a8d8c502b1a0a4b3124aff5c04c5d519af601c105220dd41ccc",
        "4ef7c5370798e1bd7a972572d12a64e4c4463d6ed04dab404ccb55867a3b1e9e",
        _combine_sampled_and_draft,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_post_update_kernel",
        "716d50004e13452fa8fefb211d3e5ca66756f7d78b9f63fda008431b3d7b0286",
        "3097fa96b4547ccdf952985b5aa933d9b73f651f919b2124025d337934273174",
        _post_update,
        ("num_warps",),
    ),
    (
        "vllm.v1.worker.gpu.structured_outputs",
        "_apply_grammar_bitmask_kernel",
        "e334b50d21b3e3f8c39e7e64ade9135b25e4039d460f6ca271e6b44335953b21",
        "fb602ffdcdbdcaf4fdb3dee73b58435e5106e107c2fd037e88b3c5cc8c75563b",
        _grammar_bitmask,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logit_bias",
        "_bias_kernel",
        "a37b8836ad5f45654eabba292a4abe7e1f37da9695e253aab79afd07530bf6bf",
        "9681ff4dfb8298a8407b5f8655ffdbb129f4474bea20bd0292ea789d4194fba9",
        _logit_bias,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logprob",
        "_fill_logprob_token_ids_kernel",
        "67b978589645faf3ad20c43fd78c6ba91d9a1ddc85623c07df8501a317c4e2bb",
        "9aa0a7b30e841c7e5137fcca3d96ad6ee5a2ef04a13166d0a940980e9fccbdf4",
        _fill_logprob_token_ids,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_prepare_prefill_inputs_kernel",
        "928ffeb93a37d7de91cea22760440a77a5af2d25f2bed4671ca0d41984a6ff35",
        "77c7c4a386f60887453fb464c275933bbcb05f79e68be455f5b0ce0012eda6ed",
        _prepare_prefill_inputs,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_prepare_pos_seq_lens_kernel",
        "5e42c1095fe391474d4cac66c76dba29259a62ff2a20ce5cf0ce5df7d52accef",
        "41458ec9cbc4829bfd42e070efdc5be7d1c6fd0a884d674324a6d0496296a096",
        _prepare_pos_seq_lens,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_get_num_sampled_and_rejected_kernel",
        "8e76f6adf25e74d7d7bdf053d50399861f50613fb707acdd0b869b1c3be82b07",
        "8172888d9239019811f577e2b0dbafcc7d656804f3503c753016f34d3cbafce1",
        _get_num_sampled_and_rejected,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_post_update_num_computed_tokens_kernel",
        "9c9754a3c48a493290d6917370a96d79620239809fc0edb08e6906cf45acfcad",
        "042f8815ef34ea47ce1c367ab3b183c34001b97d0b598907a1d5090227691a3a",
        _post_update_num_computed_tokens,
        (),
    ),
    (
        "vllm.v1.worker.gpu.input_batch",
        "_expand_idx_mapping_kernel",
        "c716f4fadb06c80d28adbb03307d989bf62f7ef9ae6694549650fdc2a7983045",
        "a91a97bdb0a7d9f6ad5095f72509905e256570d914a116741ad0c93ee51d4128",
        _expand_idx_mapping,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.bad_words",
        "_bad_words_kernel",
        "d58260c941d0deaaa6e474f18c6db2cc5b3bd959a63b9d5ee35450320ecdb55a",
        "e275dcc38c8769bebb9322fd473062d0c9319963f8b1ed0ee4c8f0e093388a54",
        _bad_words,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logprob",
        "_topk_log_softmax_kernel",
        "31c228669aa02373e285e4e8f318a4b62beda5bacbe9be3dc4e1f81715dfd908",
        "ed8ecd4bb2608522dcd02fd61f20aebb36fd6fc99eb3b6322392ea0082cf7c31",
        _topk_log_softmax,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.logprob",
        "_ranks_kernel",
        "f1b2b5dc4deef9f30924304f7bf98393138b883dd98db60f65f139ffc26e7c90",
        "1b8570353f9fe64bae9cfd8ea5fc80c5b89aa72d711adfea0d464204ccb8e1a6",
        _ranks,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.min_p",
        "_min_p_kernel",
        "07cfd4634f986ac5cd6e0f5c7f66b678c7bb6b1168cae50fd699c23fee949f0a",
        "334bc27604309735887a01c28b4ffbf2283780b45cd95f50f98cd9095022cb79",
        _min_p,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.penalties",
        "_penalties_kernel",
        "7d427188e1874917ad43b0a843b718518ef9bc4116be8367c06fe72bfba5e3dc",
        "332996dd6a4f16f92ec084fe2623bc06f3b65c46f2a667ba841505284d27fb4f",
        _penalties,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.penalties",
        "_bincount_kernel",
        "3009661a6db97446184fb9ed6458069c580a12c4a9e2cc643dff35ca4a80369c",
        "14a55431028882a62010aa7273ff46305b8f4b361067479c30b7c885367eaf5f",
        _bincount,
        (),
    ),
    (
        "vllm.v1.worker.gpu.sample.prompt_logprob",
        "_prompt_logprobs_token_ids_kernel",
        "85a3b2e004343807a21da4eb84d09548c3bb1f26d5a0db01cfb826cd7a9f3b54",
        "a711807c2084cbfea9cd3094ce73d6023f335fa65ed111c53317ab26276ca909",
        _prompt_logprob_token_ids,
        (),
    ),
)


def register_vllm_kernels() -> None:
    """Register vLLM kernels validated against the pinned source revision."""
    registry = get_registry()
    for module_name, name, source_hash, signature_hash, adapter, metadata in _KERNELS:
        module = importlib.import_module(module_name)
        kernel = getattr(module, name)
        registry.register(
            kernel,
            adapter,
            expected_source_hash=source_hash,
            expected_signature_hash=signature_hash,
            allowed_metadata=metadata,
            owner=f"torch_mcpu@{_VLLM_VERSION}",
        )
