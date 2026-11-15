# SPDX-License-Identifier: Apache-2.0
"""Fake Triton adapters for the DeepSeek-V4.1 metadata kernels.

These run outside the model graph, once per step, on the reused upstream
metadata builders. Each adapter re-expresses its kernel with ATen ops on the
launch stream; the reviewed kernel fingerprints live with their contracts in
``upstream.py``.
"""

from collections.abc import Callable
from typing import Any

import torch

from vllm_xcpu_plugin.fake_triton.runtime import (
    FakeJITFunction,
    InvalidLaunchError,
    KernelLaunch,
    get_registry,
)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidLaunchError(message)


def _i64(n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int64, device=device)


def _compute_swa_indices_and_lens(launch: KernelLaunch) -> None:
    import torch_xcpu

    args = launch.arguments
    _expect(not args["HAS_IMAGE"], "XCPU DSV4.1 SWA indices have no image spans")
    (num_tokens,) = launch.grid
    if num_tokens == 0:
        return
    torch_xcpu.ops.deepseek_v41_compute_swa_metadata(
        args["swa_indices_ptr"],
        args["swa_lens_ptr"],
        args["query_start_loc_ptr"],
        args["seq_lens_ptr"],
        args["token_to_req_indices_ptr"],
        args["is_valid_token_ptr"],
        args["block_table_ptr"],
        args["replay_start_ptr"],
        args["swa_indices_stride"],
        args["window_size"],
        args["index_width"],
        args["block_size"],
        num_tokens,
        args["token_offset"],
        False,
    )


def _compute_dspark_noncausal_swa_indices(launch: KernelLaunch) -> None:
    import torch_xcpu

    args = launch.arguments
    (num_tokens,) = launch.grid
    if num_tokens == 0:
        return
    torch_xcpu.ops.deepseek_v41_compute_swa_metadata(
        args["swa_indices_ptr"],
        args["swa_lens_ptr"],
        args["query_start_loc_ptr"],
        args["seq_lens_ptr"],
        args["token_to_req_indices_ptr"],
        args["is_valid_token_ptr"],
        args["block_table_ptr"],
        None,
        args["swa_indices_stride"],
        args["window_size"],
        args["index_width"],
        args["block_size"],
        num_tokens,
        args["token_offset"],
        True,
    )


def _compute_prefill_metadata(launch: KernelLaunch) -> None:
    import torch_xcpu

    args = launch.arguments
    torch_xcpu.ops.deepseek_v41_compute_prefill_metadata(
        args["prefill_gather_lens_ptr"],
        args["seq_lens_ptr"],
        args["query_start_loc_ptr"],
        args["replay_start_ptr"],
        args["num_prefills"],
        args["num_decodes"],
        args["window_size"],
    )


def _compressed_slot_mapping(launch: KernelLaunch) -> None:
    import torch_xcpu

    args = launch.arguments
    _expect(
        args["dcp_world_size"] == 1 and args["cp_interleave"] == 1,
        "XCPU DSV4.1 compressed slot mapping has no decode context parallel",
    )
    (num_reqs,) = launch.grid
    torch_xcpu.ops.deepseek_v41_compressed_slot_mapping(
        args["compressed_slot_mapping_ptr"],
        args["slot_mapping_ptr"],
        args["query_start_loc_ptr"],
        args["seq_lens_ptr"],
        args["block_table_ptr"],
        num_reqs,
        args["block_size"],
        args["COMPRESS_RATIO"],
        args["PAD_ID"],
    )


def _ring_slot_mapping(launch: KernelLaunch) -> None:
    import torch_xcpu

    args = launch.arguments
    torch_xcpu.ops.deepseek_v41_ring_slot_mapping(
        args["slot_mapping_ptr"],
        args["token_to_req_ptr"],
        args["block_table_ptr"],
        args["positions_ptr"],
        args["block_table_stride"],
        args["num_actual_tokens"],
        args["num_tokens"],
        args["CAPACITY"],
    )


def _gather_lookback(launch: KernelLaunch) -> None:
    import torch_xcpu

    args = launch.arguments
    (rows,) = launch.grid
    torch_xcpu.ops.deepseek_v41_gather_lookback(
        args["lookback_ptr"],
        args["idx_mapping_ptr"],
        args["num_computed_tokens_ptr"],
        args["all_token_ids_ptr"],
        args["all_token_ids_stride"],
        rows,
        args["num_reqs"],
        args["DEPTH"],
    )


def _pad_replayed_slots(launch: KernelLaunch) -> None:
    args = launch.arguments
    (num_reqs,) = launch.grid
    if num_reqs == 0:
        return
    qsl = args["query_start_loc_ptr"].to(torch.int64)
    num_tokens = int(qsl[num_reqs])
    if num_tokens == 0:
        return
    device = qsl.device
    token = _i64(num_tokens, device)
    req = torch.searchsorted(qsl[1 : num_reqs + 1].contiguous(), token, right=True)
    start = args["replay_start_ptr"].to(torch.int64)[req]
    pos = args["positions_ptr"][:num_tokens].to(torch.int64)
    replayed = (start != 0) & (pos >= start) & (pos < start + args["window"])
    slots = args["slot_mappings_ptr"]
    _expect(slots.stride(0) == args["group_stride"], "slot mapping stride mismatch")
    for group in args["cacheable_groups_ptr"][: args["NUM_GROUPS"]].tolist():
        row = slots[group, :num_tokens]
        row.copy_(torch.where(replayed, args["pad_slot_id"], row))


def _gather_replay_batch(launch: KernelLaunch) -> None:
    args = launch.arguments
    (num_reqs,) = launch.grid
    qsl = args["query_start_loc_ptr"][: num_reqs + 1].to(torch.int64).tolist()
    dropped = args["dropped_before_ptr"][: num_reqs + 1].to(torch.int64).tolist()
    seq_lens = args["seq_lens_ptr"]
    replay_start = args["replay_start_ptr"]
    window = args["window"]
    rows_out = args["rows_ptr"]
    kept_qsl = args["kept_query_start_loc_ptr"]
    kept_kv_start = args["kept_kv_start_ptr"]
    num_groups = args["NUM_GROUPS"]
    src = args["slot_mappings_ptr"]
    dst = args["kept_slot_mappings_ptr"]
    _expect(
        src.stride(0) == args["slot_mappings_stride"]
        and dst.stride(0) == args["kept_slot_mappings_stride"],
        "replay slot mapping stride mismatch",
    )
    device = rows_out.device
    starts, row_parts = [qsl[0] - dropped[0]], []
    for req in range(num_reqs):
        begin, end = qsl[req], qsl[req + 1]
        kept_begin, kept_end = begin - dropped[req], end - dropped[req + 1]
        starts.append(kept_end)
        kv_start = replay_start[req : req + 1].to(torch.int64)
        if kept_end - kept_begin < end - begin:
            kv_start = torch.maximum(
                kv_start, seq_lens[req : req + 1].to(torch.int64) - window
            )
        kept_kv_start[req : req + 1] = kv_start.to(kept_kv_start.dtype)
        row_parts.append(
            _i64(kept_end - kept_begin, device) + end - (kept_end - kept_begin)
        )
    kept_qsl[: num_reqs + 1] = torch.tensor(starts, dtype=kept_qsl.dtype, device=device)
    if not row_parts:
        return
    rows = torch.cat(row_parts)
    begin = starts[0]
    num_kept = rows.numel()
    rows_out[begin : begin + num_kept] = rows.to(rows_out.dtype)
    kept_positions = args["kept_positions_ptr"]
    kept_positions[begin : begin + num_kept] = args["positions_ptr"][rows].to(
        kept_positions.dtype
    )
    for g in range(num_groups):
        dst[g, begin : begin + num_kept] = src[g, rows]


def _engram_select_rows(launch: KernelLaunch) -> None:
    """Copy one token window out of a rank-major gathered engram buffer:
    output[t, c] = gathered[c // LOCAL_WIDTH][token_start + t][c % LOCAL_WIDTH],
    zero past ``num_tokens``."""
    import torch_xcpu

    args = launch.arguments
    gathered, output = args["gathered"], args["output"]
    num_tokens, token_start = args["num_tokens"], args["token_start"]
    local_width, width = args["LOCAL_WIDTH"], args["WIDTH"]
    _expect(args["num_elements"] == output.numel(), "engram select size mismatch")
    _expect(width % local_width == 0, "engram width is not a multiple of local width")
    rows = output.numel() // width
    if rows == 0:
        return
    torch_xcpu.ops.deepseek_v41_engram_select_rows(
        output.view(rows, width), gathered, num_tokens, token_start, local_width
    )


# (module, kernel qualname, adapter)
_ADAPTERS: tuple[tuple[str, str, Callable[[KernelLaunch], None]], ...] = (
    (
        "vllm.v1.attention.backends.mla.sparse_swa",
        "_compute_swa_indices_and_lens_kernel",
        _compute_swa_indices_and_lens,
    ),
    (
        "vllm.v1.attention.backends.mla.sparse_swa",
        "ComputeDSparkNoncausalSWAIndicesKernel.kernel",
        _compute_dspark_noncausal_swa_indices,
    ),
    (
        "vllm.v1.attention.backends.mla.sparse_swa",
        "ComputePrefillMetadataKernel.kernel",
        _compute_prefill_metadata,
    ),
    (
        "vllm.v1.attention.backends.mla.compressor_utils",
        "CompressedSlotMappingKernel.kernel",
        _compressed_slot_mapping,
    ),
    (
        "vllm.models.deepseek_v41.compressor",
        "_ring_slot_mapping_kernel",
        _ring_slot_mapping,
    ),
    (
        "vllm.models.deepseek_v41.nvidia.model_state",
        "_gather_lookback_kernel",
        _gather_lookback,
    ),
    (
        "vllm.models.deepseek_v41.nvidia.model_state",
        "_pad_replayed_slots_kernel",
        _pad_replayed_slots,
    ),
    (
        "vllm.models.deepseek_v41.nvidia.model_state",
        "_gather_replay_batch_kernel",
        _gather_replay_batch,
    ),
    (
        "vllm.models.deepseek_v41.common.engram",
        "_engram_select_rows_kernel",
        _engram_select_rows,
    ),
)


def _resolve(module_name: str, qualname: str) -> FakeJITFunction:
    import importlib

    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def register_deepseek_v41_kernels() -> None:
    """Register the adapters with the pins of their manifest patches."""
    from .upstream import MANIFEST

    pins = {
        ref.target: (ref, patch) for patch in MANIFEST.patches for ref in patch.upstream
    }
    registry = get_registry()
    for module_name, qualname, adapter in _ADAPTERS:
        kernel = _resolve(module_name, qualname)
        ref, patch = pins[f"{module_name}:{qualname}"]
        assert ref.source_hash is not None and ref.signature_hash is not None
        registry.register(
            kernel,
            adapter,
            expected_source_hash=ref.source_hash,
            expected_signature_hash=ref.signature_hash,
            allowed_metadata=("num_warps",),
            owner=f"vllm_xcpu_plugin.models.deepseek_v41 ({patch.id})",
            source_version=str(patch.source),
            defer_version_check=True,
        )
