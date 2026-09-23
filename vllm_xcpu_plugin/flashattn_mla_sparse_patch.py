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


def _xcpu_fused_mla_rope_kvcache_supported(self) -> bool:
    """The patched sparse MLA backend owns exactly this cache-update path."""
    return True


def _xcpu_fused_mla_rope_qproj_kvcache_supported(self) -> bool:
    """Fused rope + q up-projection + concat + cache update."""
    return self.kv_cache_dtype in ("auto", "bfloat16", "fp8_ds_mla")


def _xcpu_do_fused_mla_rope_qproj_kvcache_update(
    self,
    q: torch.Tensor,
    w_uk_t: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    out_q: torch.Tensor,
) -> None:
    # Shape and layout support is intentionally enforced by the operator. If
    # this semantic path is selected but the kernel lacks shape coverage, fail
    # loudly so the missing operator support is visible to developers.
    del self, k_scale
    import torch_xcpu

    torch_xcpu.ops.fused_mla_rope_qproj_cat_cache(
        q,
        w_uk_t,
        k_pe.squeeze(1),
        kv_c_normed,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        out_q,
        "auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
        is_neox,
    )


def _xcpu_do_fused_mla_rope_kvcache_update(
    self,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> None:
    # Shape and layout support is intentionally enforced by the operator. If
    # this semantic path is selected but the kernel lacks shape coverage, fail
    # loudly so the missing operator support is visible to developers.
    del self, k_scale
    import torch_xcpu

    torch_xcpu.ops.fused_mla_rope_cache(
        q_pe,
        k_pe.squeeze(1),
        kv_c_normed,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        "auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
        is_neox,
    )


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
        # [blocks, per_page_size] Bytes
        # Per Page:
        # Unquantized: page_size * (512 * BF16 + 64 * BF16 RoPE);
        #              per_page_size = page_size * 576 * 2 Bytes
        # FP8 Layout1: page_size * (512 * FP8, 4 * FP32 Scale, 64 * BF16 Rope)
        #              per_page_size = page_size * 656 Bytes
        # FP8 Layout2: page_size * (512 * FP8, 64 * BF16 Rope);
        #            & page_size * 4 * FP32 Scale
        #              per_page_size = page_size * 656 Bytes
        kv_cache,
        slot_mapping.flatten(),
        kv_cache_dtype="auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
    )


def _xcpu_forward_mqa(
    self,
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: FlashAttnMLASparseMetadata,
    layer: AttentionLayer,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(q, tuple):
        q_nope, q_rope = q
        # [tokens, heads, kv_lora_rank + qk_rope]
        query = torch.cat((q_nope, q_rope), dim=-1)
    else:
        # Already the final query emitted by the fused rope + up-projection +
        # concat op; [tokens, heads, kv_lora_rank + qk_rope].
        query = q
    output = torch.empty(
        (query.shape[0], query.shape[1], self.kv_lora_rank),
        dtype=query.dtype,
        device=query.device,
    )  # [tokens, heads, kv_lora_rank]
    logical_topk = None
    if self.is_sparse:
        assert self.topk_indices_buffer is not None
        logical_topk = self.topk_indices_buffer[: query.shape[0]]

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
        import torch_xcpu

        # The legacy kernel supports row stride, but requires unit column stride.
        if logical_topk is not None:
            logical_topk = logical_topk.contiguous()
        torch_xcpu.ops.unified_attention(
            q=query,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
            kv=kv_c_and_k_pe_cache,
            out=output,  # [tokens, q_num_heads, kv_lora_rank]
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            softmax_scale=self.scale,
            window_size=0,
            block_table=attn_metadata.block_table,
            logical_topk=logical_topk,
            kv_cache_dtype=(
                "auto" if self.kv_cache_dtype == "bfloat16" else self.kv_cache_dtype
            ),
        )
    return output, None


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
    impl_cls.fused_mla_rope_kvcache_supported = _xcpu_fused_mla_rope_kvcache_supported
    impl_cls.fused_mla_rope_qproj_kvcache_supported = (
        _xcpu_fused_mla_rope_qproj_kvcache_supported
    )
    impl_cls.do_fused_mla_rope_qproj_kvcache_update = (
        _xcpu_do_fused_mla_rope_qproj_kvcache_update
    )
    impl_cls.do_fused_mla_rope_kvcache_update = _xcpu_do_fused_mla_rope_kvcache_update
    impl_cls.do_kv_cache_update = _xcpu_do_kv_cache_update
    impl_cls.forward_mqa = _xcpu_forward_mqa
    impl_cls._xcpu_flashattn_mla_sparse_patched = True
