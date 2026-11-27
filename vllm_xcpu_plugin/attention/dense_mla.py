# SPDX-License-Identifier: Apache-2.0
"""XCPU dense MLA backend using v0.31 cache specs and unified_attention."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import MLACommonBaseImpl
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from vllm_xcpu_plugin.attention.sparse_mla import (
    XcpuMLASparseBackend,
    _xcpu_do_kv_cache_update,
)


@dataclass
class XcpuMLADenseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    num_decode_tokens: int = 0


def _validate_dense_mla_config(config: Any) -> None:
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1:
        raise NotImplementedError("XCPU dense MLA requires DCP=1")
    if parallel.prefill_context_parallel_size != 1:
        raise NotImplementedError("XCPU dense MLA requires PCP=1")
    if config.attention_config.hisparse_config is not None:
        raise NotImplementedError("XCPU dense MLA does not support HiSparse")
    spec = config.speculative_config
    if spec is not None and spec.method != "dflash":
        raise NotImplementedError(
            "XCPU v0.31 dense MLA supports only DFlash speculative decoding, "
            f"got {spec.method!r}"
        )
    if parallel.num_ubatches > 1:
        raise NotImplementedError("XCPU dense MLA does not support ubatching/DBO")
    if config.model_config.dtype != torch.bfloat16:
        raise NotImplementedError("XCPU dense MLA requires BF16 queries")


def _xcpu_dense_forward_mqa(
    self: "XcpuMLADenseImpl",
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: XcpuMLADenseMetadata,
    layer: AttentionLayer,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del layer
    import torch_xcpu

    if isinstance(q, tuple):
        q_nope, q_rope = q
        query = torch.cat((q_nope, q_rope), dim=-1)
    else:
        query = q

    output = torch.empty(
        (query.shape[0], query.shape[1], self.kv_lora_rank),
        dtype=query.dtype,
        device=query.device,
    )
    torch_xcpu.ops.unified_attention(
        q=query,
        kv=kv_c_and_k_pe_cache,
        out=output,
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=attn_metadata.seq_lens,
        softmax_scale=self.scale,
        window_size=0,
        block_table=attn_metadata.block_table,
        logical_topk=None,
        kv_cache_dtype=(
            "auto" if self.kv_cache_dtype == "bfloat16" else self.kv_cache_dtype
        ),
    )
    return output, None


class XcpuMLADenseBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = XcpuMLASparseBackend.supported_kv_cache_dtypes

    @classmethod
    def canonicalize_kv_cache_dtype(cls, kv_cache_dtype):
        return XcpuMLASparseBackend.canonicalize_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def customize_spec(cls, spec):
        return XcpuMLASparseBackend.customize_spec(spec)

    @staticmethod
    def get_name() -> str:
        return "XCPU_MLA_DENSE"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        del kv_cache_spec
        return [64, 128]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls():
        return XcpuMLADenseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return XcpuMLADenseImpl


class XcpuMLADenseMetadataBuilder(AttentionMetadataBuilder[XcpuMLADenseMetadata]):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        _validate_dense_mla_config(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XcpuMLADenseMetadata:
        del common_prefix_len, fast_build
        _, _, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=1,
        )
        return XcpuMLADenseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            num_decode_tokens=num_decode_tokens,
        )


class XcpuMLADenseImpl(MLACommonBaseImpl[XcpuMLADenseMetadata]):
    is_sparse = False
    supports_dense_mha_prefill = False
    supports_quant_query_input = False
    lse_base_on_e = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        del attn_type, q_lora_rank, indexer, q_pad_num_heads
        config = get_current_vllm_config()
        _validate_dense_mla_config(config)
        if any(
            v is not None
            for v in (
                alibi_slopes,
                sliding_window,
                logits_soft_cap,
                kv_sharing_target_layer_name,
            )
        ):
            raise NotImplementedError(
                "XCPU dense MLA supports ordinary causal attention only"
            )
        if kv_cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla", "int8_ds_mla"):
            raise NotImplementedError(
                f"Unsupported XCPU dense MLA cache dtype: {kv_cache_dtype!r}"
            )
        if (
            kv_lora_rank != 512
            or qk_rope_head_dim != 64
            or qk_nope_head_dim % 64 != 0
            or v_head_dim % 32 != 0
        ):
            raise NotImplementedError(
                "XCPU dense MLA requires kv_lora_rank=512, qk_rope_head_dim=64, "
                f"qk_nope_head_dim%64==0, v_head_dim%32==0; got "
                f"({kv_lora_rank}, {qk_rope_head_dim}, {qk_nope_head_dim}, "
                f"{v_head_dim})"
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
        )
        self.dcp_world_size = 1
        self.pcp_world_size = config.parallel_config.prefill_context_parallel_size
        # Dense MQA consumes decomposed W_UK_T / W_UV, never packed kv_b_proj.
        self.kv_b_proj._xcpu_mla_weight_only = True  # type: ignore[attr-defined]

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype

    def record_logical_topk_ready(self) -> None:
        pass

    do_kv_cache_update = _xcpu_do_kv_cache_update
    forward_mqa = _xcpu_dense_forward_mqa
