# SPDX-License-Identifier: Apache-2.0
"""GDN metadata builder replacement backed by torch_xcpu C++ loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_ORIGINAL_GDN_METADATA_BUILD: Callable[..., GDNAttentionMetadata] | None = None
# Resolved once in maybe_patch_gdn_metadata(), off the per-step path.
_BUILD_GDN_METADATA: Callable[..., tuple] | None = None
_FLA_CHUNK_SIZE = 0


@dataclass
class XcpuGDNAttentionMetadata(GDNAttentionMetadata):
    # Immutable per-build CPU scheduling facts, passed as a runtime graph input.
    xcpu_core_counts: torch.Tensor | None = None
    xcpu_core_tensors: tuple[torch.Tensor, ...] = ()


def _core_tensors(
    device: torch.device, *tensors: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    # Guardless AOT replay must fetch every slot on every invocation, including
    # slots that were absent during tracing. Distinct views also prevent Dynamo
    # from specializing two slots to the same Python Tensor object. The C++
    # builder returns the same slots; this rebuilds them after FULL staging.
    return tuple(
        tensor.view(tensor.shape)
        if tensor is not None
        else torch.empty(0, dtype=dtype, device=device)
        for tensor, dtype in zip(
            tensors,
            (
                torch.bool,
                torch.int32,
                torch.int32,
                torch.int32,
                torch.int32,
                torch.int64,
                torch.int64,
                torch.int32,
                torch.int32,
                torch.int32,
                torch.bool,
            ),
        )
    )


def _xcpu_gdn_metadata_build(
    self: GDNAttentionMetadataBuilder,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
) -> GDNAttentionMetadata:
    m = common_attn_metadata
    query_start_loc = m.query_start_loc
    if (
        query_start_loc.device.type not in ("mcpu", "privateuseone")
        or self.gdn_prefill_backend == "cutedsl"
    ):
        assert _ORIGINAL_GDN_METADATA_BUILD is not None
        return _ORIGINAL_GDN_METADATA_BUILD(
            self,
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )

    # Match upstream build(): precomputed indices are already aligned. The
    # checkpoint builder below must still receive the original block table.
    aligned_state_indices = getattr(self, "mamba_aligned_state_indices", None)
    if aligned_state_indices is not None:
        block_table_tensor = aligned_state_indices[: m.num_reqs]
        block_size_to_align = 0
    else:
        block_table_tensor = m.block_table_tensor
        block_size_to_align = (
            self.kv_cache_spec.block_size
            if self.vllm_config.cache_config.mamba_cache_mode == "align"
            else 0
        )

    # One host call covers every batch shape (prefill, decode, spec decode and
    # their mixes); see torch_xcpu/csrc/gdn_metadata.cpp for the result order.
    assert _BUILD_GDN_METADATA is not None
    tensors, counts, request_rows = _BUILD_GDN_METADATA(
        query_start_loc,
        m.query_start_loc_cpu,
        m.seq_lens,
        block_table_tensor,
        num_accepted_tokens,
        num_decode_draft_tokens_cpu,
        m.seq_lens_cpu_upper_bound,
        m.is_prefilling,
        self.num_spec if self.use_spec_decode else 0,
        m.num_actual_tokens,
        block_size_to_align,
        _FLA_CHUNK_SIZE,
    )
    (
        num_prefills,
        num_prefill_tokens,
        num_decodes,
        num_decode_tokens,
        num_spec_decodes,
        num_spec_decode_tokens,
        uniform_spec_sequence_length,
    ) = counts
    (
        has_initial_state,
        spec_query_start_loc,
        non_spec_query_start_loc,
        spec_state_indices_tensor,
        non_spec_state_indices_tensor,
        spec_sequence_masks,
        spec_token_indx,
        non_spec_token_indx,
        num_accepted_tokens,
        chunk_indices,
        chunk_offsets,
        prefill_query_start_loc,
        prefill_state_indices,
        prefill_has_initial_state,
        nums,
        mlist,
        offsetlist,
        batch_ptr,
        token_chunk_offset_ptr,
        spec_sequence_masks_cpu,
        core_counts,
    ) = tensors[:21]
    core_tensors = tuple(tensors[21:])

    nums_dict = None
    checkpoint = None
    if num_prefills > 0:
        assert mlist is not None
        nums_dict = {
            8: {
                "nums": nums,
                "tot": mlist.numel(),
                "mlist": mlist,
                "mlist_len": mlist.numel(),
                "offsetlist": offsetlist,
                "batch_ptr": batch_ptr,
                "token_chunk_offset_ptr": token_chunk_offset_ptr,
            }
        }
        checkpoint = self.checkpoint_builder.build(
            m, request_rows if num_spec_decodes > 0 else list(range(m.num_reqs))
        )

    # Function code counted on either presency non-spec decode or spec decode,
    # but not both.
    assert not (num_decodes > 0 and num_spec_decodes > 0), (
        f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
    )

    # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
    # token-padded for FULL graph replay, but the GDN state/query/accepted
    # metadata below is indexed by request.
    batch_size = m.num_reqs

    staged = False
    if (
        self.use_full_cuda_graph
        and num_prefills == 0
        and num_decodes == 0
        and num_spec_decodes <= self.decode_cudagraph_max_bs
        and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
    ):
        staged = True
        assert spec_sequence_masks is not None
        self.spec_state_indices_tensor[:num_spec_decodes].copy_(
            spec_state_indices_tensor, non_blocking=True
        )
        spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
        spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

        self.spec_sequence_masks[:num_spec_decodes].copy_(
            spec_sequence_masks[:num_spec_decodes], non_blocking=True
        )
        spec_sequence_masks = self.spec_sequence_masks[:batch_size]
        spec_sequence_masks[num_spec_decodes:].fill_(False)

        assert non_spec_token_indx is not None and spec_token_indx is not None
        self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
            non_spec_token_indx, non_blocking=True
        )
        non_spec_token_indx = self.non_spec_token_indx[: non_spec_token_indx.size(0)]

        self.spec_token_indx[: spec_token_indx.size(0)].copy_(
            spec_token_indx, non_blocking=True
        )
        spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

        self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
            spec_query_start_loc, non_blocking=True
        )
        spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
        spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
        spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

        self.num_accepted_tokens[:num_spec_decodes].copy_(
            num_accepted_tokens, non_blocking=True
        )
        num_accepted_tokens = self.num_accepted_tokens[:batch_size]
        num_accepted_tokens[num_spec_decodes:].fill_(1)

    if (
        self.use_full_cuda_graph
        and num_prefills == 0
        and num_spec_decodes == 0
        and num_decodes <= self.decode_cudagraph_max_bs
    ):
        staged = True
        self.non_spec_state_indices_tensor[:num_decodes].copy_(
            non_spec_state_indices_tensor, non_blocking=True
        )
        non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:batch_size]
        non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

        self.non_spec_query_start_loc[: num_decodes + 1].copy_(
            non_spec_query_start_loc, non_blocking=True
        )
        non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
        non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
        non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

    if staged:
        core_tensors = _core_tensors(
            query_start_loc.device,
            has_initial_state,
            spec_query_start_loc,
            non_spec_query_start_loc,
            spec_state_indices_tensor,
            non_spec_state_indices_tensor,
            spec_token_indx,
            non_spec_token_indx,
            num_accepted_tokens,
            prefill_query_start_loc,
            prefill_state_indices,
            prefill_has_initial_state,
        )
    attn_metadata = XcpuGDNAttentionMetadata(
        xcpu_core_tensors=core_tensors,
        xcpu_core_counts=core_counts,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_actual_tokens=m.num_actual_tokens,
        checkpoint=checkpoint,
        has_initial_state=has_initial_state,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=prefill_query_start_loc,
        prefill_state_indices=prefill_state_indices,
        prefill_has_initial_state=prefill_has_initial_state,
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=non_spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_sequence_masks=spec_sequence_masks,
        spec_sequence_masks_cpu=spec_sequence_masks_cpu,
        uniform_spec_sequence_length=uniform_spec_sequence_length or None,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    return attn_metadata


def maybe_patch_gdn_metadata() -> None:
    global _ORIGINAL_GDN_METADATA_BUILD, _BUILD_GDN_METADATA, _FLA_CHUNK_SIZE

    if _ORIGINAL_GDN_METADATA_BUILD is not None:
        return

    import torch_xcpu
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("gdn_metadata",))
    _BUILD_GDN_METADATA = torch_xcpu._C.build_gdn_metadata
    _FLA_CHUNK_SIZE = FLA_CHUNK_SIZE
    _ORIGINAL_GDN_METADATA_BUILD = GDNAttentionMetadataBuilder.build
    GDNAttentionMetadataBuilder.build = wraps(_ORIGINAL_GDN_METADATA_BUILD)(  # type: ignore[method-assign]
        _xcpu_gdn_metadata_build
    )
