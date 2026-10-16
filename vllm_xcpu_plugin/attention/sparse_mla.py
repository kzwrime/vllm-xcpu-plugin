# SPDX-License-Identifier: Apache-2.0
"""XCPU sparse MLA using v0.31 cache specs and metadata ownership."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
)

from vllm_xcpu_plugin import shared_kv_cache


@dataclass
class XcpuMLASparseMetadata(SparseMLACommonMetadata):
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

    shared = shared_kv_cache.runtime()
    if shared is not None and not shared.writer:
        shared_kv_cache.publish(kv_cache)
        return
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

    if shared is not None:
        shared_kv_cache.publish(kv_cache)


def _xcpu_forward_mqa(
    self,
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: XcpuMLASparseMetadata,
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


def validate_sparse_config(config):
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1:
        raise NotImplementedError("XCPU sparse MLA requires DCP=1")
    if config.attention_config.hisparse_config is not None:
        raise NotImplementedError("XCPU sparse MLA does not support HiSparse")
    if (
        config.speculative_config is not None
        and config.speculative_config.method not in ("dflash", "dflash2")
    ):
        raise NotImplementedError("XCPU sparse MLA supports DFlash speculation")
    if parallel.num_ubatches > 1:
        raise NotImplementedError("XCPU sparse MLA does not support ubatching/DBO")
    if config.model_config.dtype != torch.bfloat16:
        raise NotImplementedError("XCPU sparse MLA requires BF16 queries")


class XcpuMLASparseBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_ds_mla"]

    @classmethod
    def canonicalize_kv_cache_dtype(cls, kv_cache_dtype):
        return "fp8_ds_mla" if kv_cache_dtype in ("fp8", "fp8_e4m3") else kv_cache_dtype

    @staticmethod
    def get_name():
        return "XCPU_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [64, 128]

    @classmethod
    def get_supported_head_sizes(cls):
        return [576]

    @classmethod
    def is_mla(cls):
        return True

    @classmethod
    def is_sparse(cls):
        return True

    @staticmethod
    def get_builder_cls():
        return XcpuMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return XcpuMLASparseImpl


class XcpuMLASparseMetadataBuilder(SparseMLACommonMetadataBuilder):
    metadata_cls = XcpuMLASparseMetadata
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        validate_sparse_config(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.supports_draft_decode_metadata_update = False
        self._xcpu_backend = None
        if vllm_config.cache_config.cache_dtype == "fp8_ds_mla":
            import torch_xcpu

            heads = self.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            self._xcpu_backend = torch_xcpu.ops.SparseMlaFp8(heads)
            self._xcpu_buffers = self._xcpu_backend.allocate_metadata(
                vllm_config.scheduler_config.max_num_batched_tokens, device
            )

    def _build_req_id_per_token(self, common_attn_metadata):
        # XCPU kernels consume query_start_loc directly, so no CUDA physical
        # index conversion or token-to-request map is needed.
        return self.req_id_per_token_buffer[:0]

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if self._xcpu_backend is not None:
            is_prefilling = common_attn_metadata.is_prefilling
            if is_prefilling is None or is_prefilling.device.type != "cpu":
                raise ValueError("XCPU sparse MLA requires CPU is_prefilling metadata")
            metadata.xcpu_backend = self._xcpu_backend
            metadata.xcpu_schedule = self._xcpu_backend.build_metadata(
                metadata.query_start_loc,
                metadata.num_actual_tokens,
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
                has_prefill=bool(
                    is_prefilling[: common_attn_metadata.num_reqs].any().item()
                ),
                buffers=self._xcpu_buffers,
                kv_slots_hint=min(metadata.max_seq_len, metadata.topk_tokens),
            )
        return metadata


class XcpuMLASparseImpl(SparseMLACommonImpl):
    supports_dense_mha_prefill = False
    supports_quant_query_input = False

    def __init__(self, **kwargs):
        config = get_current_vllm_config()
        validate_sparse_config(config)
        if any(
            kwargs.get(key)
            for key in (
                "alibi_slopes",
                "sliding_window",
                "logits_soft_cap",
                "kv_sharing_target_layer_name",
            )
        ):
            raise NotImplementedError(
                "XCPU sparse MLA supports ordinary causal attention only"
            )
        if kwargs["kv_cache_dtype"] not in ("auto", "bfloat16", "fp8_ds_mla"):
            raise NotImplementedError("Unsupported XCPU MLA cache dtype")
        if (
            kwargs["kv_lora_rank"],
            kwargs["qk_rope_head_dim"],
            kwargs["qk_nope_head_dim"],
            kwargs["v_head_dim"],
        ) != (512, 64, 192, 256):
            raise NotImplementedError(
                "XCPU sparse MLA currently supports GLM5.2 dimensions"
            )
        super().__init__(**kwargs)
        self.dcp_world_size = 1
        self.pcp_world_size = config.parallel_config.prefill_context_parallel_size
        # Sparse-only attention consumes the canonical weight once, never the
        # packed linear representation. Set before the loader postprocesses it.
        self.kv_b_proj._xcpu_mla_weight_only = True

    def record_logical_topk_ready(self):
        # XCPU consumes logical indices directly on the producer stream. The
        # native group event is only needed for side-stream physical conversion.
        pass

    do_kv_cache_update = _xcpu_do_kv_cache_update
    forward_mqa = _xcpu_forward_mqa
