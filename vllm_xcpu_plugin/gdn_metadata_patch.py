# SPDX-License-Identifier: Apache-2.0
"""GDN metadata builder replacement backed by torch_xcpu C++ loops."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_ORIGINAL_GDN_METADATA_BUILD: Callable[..., GDNAttentionMetadata] | None = None


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
    query_start_loc_cpu = m.query_start_loc_cpu
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

    from vllm.third_party.flash_linear_attention.ops.utils import (
        FLA_CHUNK_SIZE,
    )

    # A one-token first prefill must clear recycled state; a resumed chunk can
    # use decode kernels. These are CPU scheduling facts, not device readbacks.
    no_prior_state_cpu = None
    if m.is_prefilling is not None:
        assert m.seq_lens_cpu_upper_bound is not None
        query_lens_cpu = query_start_loc_cpu.diff()
        no_prior_state_cpu = (
            (query_lens_cpu > 0)
            & (m.seq_lens_cpu_upper_bound <= query_lens_cpu)
            & m.is_prefilling
        )

    # Keep this result order in sync with torch_xcpu/csrc/gdn_metadata.cpp.
    outputs = torch.ops.torch_xcpu.allocate_gdn_metadata_outputs(
        query_start_loc,
        query_start_loc_cpu,
        num_decode_draft_tokens_cpu,
        self.num_spec if self.use_spec_decode else 0,
        m.max_query_len,
        FLA_CHUNK_SIZE,
        no_prior_state_cpu,
    )
    torch.ops.torch_xcpu.build_gdn_metadata_out(
        query_start_loc,
        query_start_loc_cpu,
        m.seq_lens,
        m.block_table_tensor,
        num_accepted_tokens,
        num_decode_draft_tokens_cpu,
        self.num_spec if self.use_spec_decode else 0,
        m.num_actual_tokens,
        m.max_query_len,
        self.kv_cache_spec.block_size
        if self.vllm_config.cache_config.mamba_cache_mode == "align"
        else 0,
        FLA_CHUNK_SIZE,
        outputs,
        no_prior_state_cpu,
    )
    *counts, present = outputs[21].tolist()
    tensors = [
        tensor if present & (1 << i) else None for i, tensor in enumerate(outputs[:19])
    ]
    num_prefills, _, num_decodes, _, num_spec_decodes, _ = counts
    if num_spec_decodes == 0:
        tensors[2] = query_start_loc
    elif num_prefills == 0 and num_decodes == 0:
        tensors[1] = query_start_loc[: num_spec_decodes + 1]
    else:
        tensors[6], tensors[7] = outputs[19:21]
    if num_prefills > 0:
        if num_spec_decodes == 0 and num_decodes > 0:
            assert tensors[4] is not None
            assert tensors[0] is not None
            tensors[12] = tensors[4][num_decodes:]
            tensors[13] = tensors[0][num_decodes:]
        else:
            tensors[11:14] = [tensors[2], tensors[4], tensors[0]]
    (
        num_prefills,
        num_prefill_tokens,
        num_decodes,
        num_decode_tokens,
        num_spec_decodes,
        num_spec_decode_tokens,
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
    ) = tensors
    nums_dict = None
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

    # Function code counted on either presency non-spec decode or spec decode,
    # but not both.
    assert not (num_decodes > 0 and num_spec_decodes > 0), (
        f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
    )

    # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
    # token-padded for FULL graph replay, but the GDN state/query/accepted
    # metadata below is indexed by request.
    batch_size = m.num_reqs

    if (
        self.use_full_cuda_graph
        and num_prefills == 0
        and num_decodes == 0
        and num_spec_decodes <= self.decode_cudagraph_max_bs
        and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
    ):
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

    attn_metadata = GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_actual_tokens=m.num_actual_tokens,
        xcpu_runtime_metadata_handle=self._xcpu_runtime_metadata_handle,
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
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    if self._xcpu_runtime_metadata_handle is not None:
        import torch_xcpu.ops_defs.gdn_decode_state  # noqa: F401

        torch.ops.torch_xcpu.set_gdn_runtime_metadata(
            self._xcpu_runtime_metadata_handle,
            num_prefills,
            num_decodes,
            num_spec_decodes,
            m.num_actual_tokens,
            num_decode_tokens,
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
    return attn_metadata


def maybe_patch_gdn_metadata() -> None:
    global _ORIGINAL_GDN_METADATA_BUILD

    if _ORIGINAL_GDN_METADATA_BUILD is not None:
        return

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("gdn_metadata",))
    _ORIGINAL_GDN_METADATA_BUILD = GDNAttentionMetadataBuilder.build
    GDNAttentionMetadataBuilder.build = wraps(_ORIGINAL_GDN_METADATA_BUILD)(  # type: ignore[method-assign]
        _xcpu_gdn_metadata_build
    )
