# SPDX-License-Identifier: Apache-2.0
"""Keep vLLM scheduling/cache ownership; bind XCPU metadata and attention together."""

from dataclasses import dataclass
from functools import wraps
from typing import Any, cast

import torch
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseMetadata,
)


@dataclass
class XcpuFlashAttnMLASparseMetadata(FlashAttnMLASparseMetadata):
    xcpu_backend: Any = None
    xcpu_schedule: Any = None


def _xcpu_do_kv_cache_update(
    self,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> None:
    if kv_cache.numel() == 0:
        return
    import torch_xcpu

    torch_xcpu.ops.reshape_and_cache(
        kv_c_normed,  # [tokens, kv_lora_rank]
        k_pe.squeeze(1),  # [tokens, qk_rope]
        # Unquantized: [blocks, page_size, rank + rope]; FP8: [..., 656] bytes
        kv_cache,
        slot_mapping.flatten(),
        kv_cache_dtype=kv_cache_dtype,
    )


def _xcpu_forward_mqa(
    self,
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: FlashAttnMLASparseMetadata,
    layer: AttentionLayer,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(q, tuple):
        raise NotImplementedError(
            "FlashAttnMLASparseImpl expects split (q_nope, q_rope)"
        )
    q_nope, q_rope = q
    # [tokens, heads, kv_lora_rank + qk_rope]
    query = torch.cat((q_nope, q_rope), dim=-1)
    output = torch.empty_like(q_nope)  # [tokens, heads, kv_lora_rank]
    logical_topk = None
    if self.is_sparse:
        assert self.topk_indices_buffer is not None, (
            "XCPU sparse MLA requires the shared topk indices buffer."
        )
        logical_topk = self.topk_indices_buffer

    # TODO: fix torch compile
    if self.kv_cache_dtype == "fp8_ds_mla":
        attn_metadata = cast(XcpuFlashAttnMLASparseMetadata, attn_metadata)
        backend = attn_metadata.xcpu_backend
        if backend is None or attn_metadata.xcpu_schedule is None:
            raise RuntimeError(
                "FP8 sparse MLA metadata must be built before layer execution"
            )
        backend.run(
            query,
            kv_c_and_k_pe_cache,
            logical_topk,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self.scale,
            output,
            attn_metadata.xcpu_schedule,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=attn_metadata.max_seq_len,
        )
    else:
        # MQA covers the leading decode rows (decode-first reorder). On a mixed
        # batch the varlen metadata must be narrowed to the decode requests:
        # with the full-batch cu_seqlens_q the kernel would walk the prefill
        # requests too and read q past its end.
        if attn_metadata.num_decodes == attn_metadata.num_reqs:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            block_table = attn_metadata.block_table
            max_seqlen_q = attn_metadata.max_query_len
        else:
            n = q.size(0)
            cu_seqlens_q = attn_metadata.query_start_loc[: n + 1]
            seqused_k = attn_metadata.seq_lens[:n]
            block_table = attn_metadata.block_table[:n]
            max_seqlen_q = max(1, int(torch.diff(cu_seqlens_q).max().item()))
    
        import torch_xcpu
    
        torch_xcpu.ops.unified_attention(
            q=q,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
            kv=kv_c_and_k_pe_cache,
            out=output,  # [tokens, q_num_heads, kv_lora_rank]
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=self.scale,
            window_size=0,
            block_table=block_table,
            logical_topk=logical_topk,
            kv_cache_dtype=self.kv_cache_dtype,
        )
    return output, None


def _xcpu_make_sparse_forward_mha(orig_forward_mha):
    def _xcpu_sparse_forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        # Bridge the shared top-k buffer into the prefill backend call.
        # Rows are ordered decode-first, so the prefill rows are
        # [num_decode_tokens, num_decode_tokens + q tokens); entries are
        # request-relative logical KV positions. The downstream
        # attn_varlen_diff_headdims op demotes to dense per request when the
        # packed KV prefix does not exceed topk_tokens.
        backend = None
        if self.topk_indices_buffer is not None and attn_metadata.prefill is not None:
            backend = getattr(attn_metadata.prefill, "prefill_backend", None)
            if backend is not None:
                num_decode_tokens = attn_metadata.num_decode_tokens
                backend._xcpu_prefill_logical_topk = self.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + q.shape[0]
                ]
        try:
            return orig_forward_mha(
                self,
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        finally:
            if backend is not None:
                backend._xcpu_prefill_logical_topk = None

    return _xcpu_sparse_forward_mha


def maybe_patch_vllm_flashattn_mla_sparse() -> None:
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
        FlashAttnMLASparseImpl,
        FlashAttnMLASparseMetadataBuilder,
    )

    impl_cls = cast(Any, FlashAttnMLASparseImpl)
    if getattr(impl_cls, "_xcpu_flashattn_mla_sparse_patched", False):
        return
    builder_cls = cast(Any, FlashAttnMLASparseMetadataBuilder)
    original_init = builder_cls.__init__
    original_build = builder_cls.build

    @wraps(original_init)
    def initialize_builder(self, kv_cache_spec, layer_names, vllm_config, device):
        original_init(self, kv_cache_spec, layer_names, vllm_config, device)
        self._xcpu_backend = None
        self._xcpu_buffers = None
        if vllm_config.cache_config.cache_dtype in (
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            import torch_xcpu

            dims = self.mla_dims
            if dims.kv_lora_rank != 512 or dims.qk_rope_head_dim != 64:
                raise NotImplementedError(
                    "XCPU FP8 sparse MLA supports GLM5.2 D576/V512"
                )
            parallel = vllm_config.parallel_config
            if (
                parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
            ):
                raise NotImplementedError("XCPU FP8 sparse MLA requires DCP=PCP=1")
            heads = self.model_config.get_num_attention_heads(parallel)
            self.metadata_cls = XcpuFlashAttnMLASparseMetadata
            self._xcpu_backend = torch_xcpu.ops.SparseMlaFp8(heads)
            self._xcpu_buffers = self._xcpu_backend.allocate_metadata(
                vllm_config.scheduler_config.max_num_batched_tokens,
                device,
            )

    @wraps(original_build)
    def build_metadata(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        if self._xcpu_backend is not None:
            # vLLM computes this CPU flag from computed_tokens < prompt_tokens.
            # Query length and num_prefills describe kernel grouping, not phase.
            is_prefilling = common_attn_metadata.is_prefilling
            if is_prefilling is None or is_prefilling.device.type != "cpu":
                raise ValueError(
                    "XCPU sparse MLA requires vLLM CPU is_prefilling metadata"
                )
            has_prefill = bool(
                is_prefilling[: common_attn_metadata.num_reqs].any().item()
            )
            metadata.xcpu_backend = self._xcpu_backend
            metadata.xcpu_schedule = self._xcpu_backend.build_metadata(
                metadata.query_start_loc,
                metadata.num_actual_tokens,
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
                has_prefill=has_prefill,
                buffers=self._xcpu_buffers,
                kv_slots_hint=min(metadata.max_seq_len, metadata.topk_tokens),
            )
        return metadata

    builder_cls.__init__ = initialize_builder
    builder_cls.build = build_metadata
    impl_cls.do_kv_cache_update = _xcpu_do_kv_cache_update
    # Note: register this when prefill backend needs to run with topk_indices_buffer
    # impl_cls.forward_mha = _xcpu_make_sparse_forward_mha(sparse_any.forward_mha)
    impl_cls.forward_mqa = _xcpu_forward_mqa
    impl_cls._xcpu_flashattn_mla_sparse_patched = True
