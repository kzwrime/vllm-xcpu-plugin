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


def _swa_window_slots(
    start_pos: torch.Tensor,
    swa_len: torch.Tensor,
    req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    width: int,
) -> torch.Tensor:
    offset = _i64(width, start_pos.device)
    pos = start_pos.unsqueeze(-1) + offset
    valid = offset < swa_len.unsqueeze(-1)
    page_col = torch.where(valid, pos // block_size, 0)
    pages = block_table.to(torch.int64)[req.unsqueeze(-1).expand_as(pos), page_col]
    return torch.where(valid, pages * block_size + pos % block_size, -1)


def _swa_rows(args: dict[str, Any], num_tokens: int) -> tuple[torch.Tensor, ...]:
    token = _i64(num_tokens, args["swa_lens_ptr"].device) + args["token_offset"]
    is_valid = args["is_valid_token_ptr"][token]
    req = args["token_to_req_indices_ptr"][token].to(torch.int64)
    qsl = args["query_start_loc_ptr"].to(torch.int64)
    query_start = qsl[req]
    query_len = qsl[req + 1] - query_start
    seq_len = args["seq_lens_ptr"].to(torch.int64)[req]
    return token, is_valid, req, query_start, query_len, seq_len


def _store_swa(
    args: dict[str, Any],
    num_tokens: int,
    is_valid: torch.Tensor,
    start_pos: torch.Tensor,
    swa_len: torch.Tensor,
    req: torch.Tensor,
) -> None:
    width = args["index_width"]
    swa_len = torch.where(is_valid, swa_len, 0)
    slots = _swa_window_slots(
        start_pos, swa_len, req, args["block_table_ptr"], args["block_size"], width
    )
    indices = args["swa_indices_ptr"]
    rows = indices.view(-1)[: indices.numel()].as_strided(
        (num_tokens, width), (args["swa_indices_stride"], 1)
    )
    rows.copy_(slots.to(indices.dtype))
    args["swa_lens_ptr"][:num_tokens] = swa_len.to(args["swa_lens_ptr"].dtype)


def _compute_swa_indices_and_lens(launch: KernelLaunch) -> None:
    args = dict(launch.arguments)
    _expect(not args["HAS_IMAGE"], "XCPU DSV4.1 SWA indices have no image spans")
    (num_tokens,) = launch.grid
    if num_tokens == 0:
        return
    token, is_valid, req, query_start, query_len, seq_len = _swa_rows(args, num_tokens)
    pos = seq_len - query_len + token - query_start
    window = args["window_size"]
    start_pos = (pos - (window - 1)).clamp_min(0)
    replay_start = args["replay_start_ptr"].to(torch.int64)[req]
    start_pos = torch.maximum(start_pos, replay_start)
    _store_swa(args, num_tokens, is_valid, start_pos, pos + 1 - start_pos, req)


def _compute_dspark_noncausal_swa_indices(launch: KernelLaunch) -> None:
    args = dict(launch.arguments)
    (num_tokens,) = launch.grid
    if num_tokens == 0:
        return
    _, is_valid, req, _, query_len, seq_len = _swa_rows(args, num_tokens)
    start_pos = (seq_len - query_len - args["window_size"]).clamp_min(0)
    _store_swa(args, num_tokens, is_valid, start_pos, seq_len - start_pos, req)


def _compute_prefill_metadata(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_prefills, num_decodes = args["num_prefills"], args["num_decodes"]
    reqs = slice(num_decodes, num_decodes + num_prefills)
    seq_len = args["seq_lens_ptr"][reqs].to(torch.int64)
    qsl = args["query_start_loc_ptr"].to(torch.int64)
    query_len = qsl[num_decodes + 1 : num_decodes + num_prefills + 1] - qsl[reqs]
    replay_start = args["replay_start_ptr"][reqs].to(torch.int64)
    visible_prefix = (seq_len - query_len - replay_start).clamp_min(0)
    gather_len = query_len + visible_prefix.clamp_max(args["window_size"] - 1)
    out = args["prefill_gather_lens_ptr"]
    out[:num_prefills] = gather_len.to(out.dtype)


def _compressed_slot_mapping(launch: KernelLaunch) -> None:
    args = launch.arguments
    _expect(
        args["dcp_world_size"] == 1 and args["cp_interleave"] == 1,
        "XCPU DSV4.1 compressed slot mapping has no decode context parallel",
    )
    ratio, pad = args["COMPRESS_RATIO"], args["PAD_ID"]
    (num_reqs,) = launch.grid
    qsl = args["query_start_loc_ptr"].to(torch.int64)
    num_tokens = int(qsl[num_reqs])
    out = args["compressed_slot_mapping_ptr"]
    if num_tokens == 0:
        return
    device = out.device
    token = _i64(num_tokens, device)
    req = torch.searchsorted(qsl[1 : num_reqs + 1].contiguous(), token, right=True)
    seq_len = args["seq_lens_ptr"].to(torch.int64)[req]
    query_len = qsl[req + 1] - qsl[req]
    pos = seq_len - query_len + token - qsl[req]
    slot = args["slot_mapping_ptr"][:num_tokens]
    valid = ((pos + 1) % ratio == 0) & (slot != pad)
    group = pos // ratio
    block_size = args["block_size"]
    page_col = torch.where(valid, group // block_size, 0)
    pages = args["block_table_ptr"].to(torch.int64)[req, page_col]
    slots = torch.where(valid, pages * block_size + group % block_size, pad)
    out[:num_tokens] = slots.to(out.dtype)


def _ring_slot_mapping(launch: KernelLaunch) -> None:
    args = launch.arguments
    num_actual, num_tokens = args["num_actual_tokens"], args["num_tokens"]
    out = args["slot_mapping_ptr"]
    device = out.device
    token = _i64(num_tokens, device)
    valid = token < num_actual
    safe = torch.where(valid, token, 0)
    req = args["token_to_req_ptr"][safe].to(torch.int64)
    block_table = args["block_table_ptr"]
    stride = args["block_table_stride"]
    first_page = block_table.view(-1)[req * stride].to(torch.int64)
    pos = args["positions_ptr"][safe].to(torch.int64)
    capacity = args["CAPACITY"]
    slot = first_page * capacity + pos % capacity
    # Block 0 is the null block: rows without a ring block must not write.
    out[:num_tokens] = torch.where(valid & (first_page != 0), slot, -1).to(out.dtype)


def _gather_lookback(launch: KernelLaunch) -> None:
    args = launch.arguments
    out = args["lookback_ptr"]
    depth, num_reqs = args["DEPTH"], args["num_reqs"]
    (rows,) = launch.grid
    device = out.device
    batch = _i64(rows, device)
    in_batch = batch < num_reqs
    req_state = torch.where(
        in_batch, args["idx_mapping_ptr"][batch.clamp_max(max(num_reqs - 1, 0))], 0
    ).to(torch.int64)
    num_computed = args["num_computed_tokens_ptr"][req_state].to(torch.int64)
    pos = num_computed.unsqueeze(-1) - 1 - _i64(depth, device)
    valid = in_batch.unsqueeze(-1) & (pos >= 0)
    table = args["all_token_ids_ptr"]
    stride = args["all_token_ids_stride"]
    flat = table.view(-1)[: table.numel()]
    ids = flat[torch.where(valid, req_state.unsqueeze(-1) * stride + pos, 0)]
    out.view(-1)[: rows * depth] = torch.where(valid, ids, -1).to(out.dtype).view(-1)


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
    args = launch.arguments
    gathered, output = args["gathered"], args["output"]
    num_tokens, token_start = args["num_tokens"], args["token_start"]
    local_width, width = args["LOCAL_WIDTH"], args["WIDTH"]
    _expect(args["num_elements"] == output.numel(), "engram select size mismatch")
    _expect(width % local_width == 0, "engram width is not a multiple of local width")
    ranks, rows = width // local_width, output.numel() // width
    out = output.view(rows, ranks, local_width)
    tokens = token_start + _i64(rows, output.device)
    valid = tokens < num_tokens
    source = gathered.view(-1)[: ranks * num_tokens * local_width].view(
        ranks, num_tokens, local_width
    )
    picked = source[:, tokens.clamp_max(max(num_tokens - 1, 0))].permute(1, 0, 2)
    out.copy_(torch.where(valid.view(rows, 1, 1), picked, 0))


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
