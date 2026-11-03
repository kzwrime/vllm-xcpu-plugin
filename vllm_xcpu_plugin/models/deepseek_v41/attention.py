# SPDX-License-Identifier: Apache-2.0
"""XCPU sparse-MLA attention layer for DeepSeek-V4.1.

Mirrors the SM100 mega-attention path layer by layer (see the module docstring
of ``ops``): MXFP8 SWA record, NVFP4 compressed record, MXFP4 indexer keys,
attention with the inverse RoPE fused in. Activations stay BF16.
"""

from typing import Any, ClassVar, cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from vllm_xcpu_plugin.layers.fp32_out_linear import XcpuFp32OutLinearMethod

from . import ops
from .backends import (
    XcpuDeepseekV41SparseMLABackend,
    XcpuDeepseekV41SWABackend,
)


def _xcpu_deepseek_v41_prepare_and_attn(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Opaque boundary around everything that reads attention metadata: the
    SWA / compressor / indexer cache writes, the indexer and the sparse MLA.

    Mirrors the eager break of the CUDA path (``_prepare_and_attn_eager``):
    the cache writes have no tensor arguments torch.compile could track, and
    the single TRACE_ONCE trace runs without metadata, which would otherwise
    bake the metadata-free branches into the graph.
    """
    self = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    self._prepare_and_attn(
        hidden_states,
        qr,
        kv,
        None,
        kv_score,
        indexer_weights,
        positions,
        attn_out,
    )


def _xcpu_deepseek_v41_prepare_and_attn_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return None


direct_register_custom_op(
    op_name="xcpu_deepseek_v41_prepare_and_attn",
    op_func=_xcpu_deepseek_v41_prepare_and_attn,
    mutates_args=["attn_out"],
    fake_impl=_xcpu_deepseek_v41_prepare_and_attn_fake,
)


class XcpuDeepseekV41IndexerOp(nn.Module):
    """Stands in for ``SparseAttnIndexer``: scores and top-k for every token
    straight off the paged MXFP4 K cache, prefill and decode alike."""

    def __init__(
        self,
        k_cache: DeepseekV4IndexerCache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor | None,
        skip_k_cache_insert: bool = True,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ) -> None:
        super().__init__()
        assert skip_k_cache_insert and use_fp4_cache
        assert topk_indices_buffer is not None
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.topk_indices_buffer = topk_indices_buffer
        self.compress_ratio = compress_ratio
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write

    def forward(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        import torch_xcpu

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer
        metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        topk_out = self.topk_indices_buffer[:, : self.topk_tokens]
        consumer = self.candidate_blocks is not None and not self.candidate_write
        for chunk, (start, end, schedule) in enumerate(metadata.chunks):
            candidate_metadata = None
            if consumer:
                # The source layer wrote this step's blocks; every consumer
                # of the step shares the metadata built from them.
                candidate_metadata = metadata.candidate_metadata.get(chunk)
                if candidate_metadata is None:
                    candidate_metadata = (
                        torch_xcpu.ops.deepseek_v41_indexer_candidate_metadata(
                            cast(torch.Tensor, self.candidate_blocks)[start:end],
                            metadata.token_to_req_indices[start:end],
                            metadata.positions[start:end],
                            self.compress_ratio,
                            self.candidate_block_size,
                            metadata.max_seq_len,
                        )
                    )
                    metadata.candidate_metadata[chunk] = candidate_metadata
            torch_xcpu.ops.deepseek_v41_indexer_topk(
                q[start:end],
                weights[start:end],
                self.k_cache.kv_cache,
                metadata.block_table,
                metadata.token_to_req_indices[start:end],
                metadata.positions[start:end],
                self.compress_ratio,
                topk_out[start:end],
                metadata.max_seq_len,
                None
                if self.candidate_blocks is None
                else self.candidate_blocks[start:end],
                self.candidate_block_size,
                self.candidate_write,
                schedule=None if consumer else schedule,
                candidate_metadata=candidate_metadata,
            )
        return self.topk_indices_buffer


class XcpuDeepseekV41Indexer(DeepseekV4Indexer):
    attn_cls: ClassVar[type] = XcpuDeepseekV41IndexerOp  # type: ignore[assignment]

    def _produce_k(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        import torch_xcpu

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or latent is None:
            return
        assert self.owns_k
        metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        k_pre, _ = self.wk(latent)
        torch_xcpu.ops.deepseek_v41_indexer_k_norm_rope_store(
            k_pre,
            positions,
            rotary_emb.cos_sin_cache,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            self.k_cache.kv_cache,
            metadata.slot_mapping,
            self.compress_ratio,
        )

    def forward_q(
        self,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        import torch_xcpu

        q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_head, self.head_dim)
        q, weights = torch_xcpu.ops.deepseek_v41_indexer_q_rope(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            self.softmax_scale,
            self.n_head**-0.5,
        )
        return q, None, weights

    def forward(
        self,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        qr_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        import torch_xcpu

        if self.owns_k:
            # The K write must land before the indexer op reads the cache.
            self._produce_k(latent, positions, rotary_emb)
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
                # Fewer candidates than top-k: every one of them is selected.
                assert self.topk_indices_buffer is not None
                num_tokens = metadata.num_decode_tokens + metadata.num_prefill_tokens
                torch_xcpu.ops.deepseek_v41_fill_short_context_topk(
                    self.topk_indices_buffer[:num_tokens],
                    positions[:num_tokens],
                    self.compress_ratio,
                )
                return None, None, None
        return self.forward_q(qr, qr_scale, indexer_weights, positions, rotary_emb)


class XcpuDeepseekV41Attention(DeepseekV4Attention):
    """DeepSeek-V4.1 attention on XCPU."""

    backend_cls = XcpuDeepseekV41SparseMLABackend
    swa_backend_cls = XcpuDeepseekV41SWABackend

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        # One MCPU stream: the CUDA path's aux-stream overlap has no
        # counterpart here, so every helper runs sequentially.
        super().__init__(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=None,
            candidate_block_buffer=candidate_block_buffer,
        )
        # The grouped wo_a runs in ``_o_proj`` off the [G*R, D] weight, so its
        # linear method must not take the batched-GEMM kernel path. The MXFP8
        # kernel prepacks it as [G, R, D] for the grouped W8A16 einsum.
        self.wo_a.is_bmm = False
        self.wo_a.xcpu_grouped_wo_a = self.n_local_groups
        # The compressor kv/score projection keeps the FP32 accumulator; its
        # BF16 weight is prepacked at load (quant_config=None upstream).
        if self.compressor is not None:
            wkv_wgate = self.compressor.fused_wkv_wgate
            assert type(wkv_wgate.quant_method) is UnquantizedLinearMethod
            wkv_wgate.quant_method = XcpuFp32OutLinearMethod()
        # The metadata-dependent middle of the upstream forward runs behind
        # the opaque op; the input projections and _o_proj stay in the graph.
        self._prepare_and_attn_fn = self._opaque_prepare_and_attn

    @property
    def packed_kv_cache_dtype(self) -> CacheDType:
        # The SM100 mega-attention record set: MXFP8 window, NVFP4 compressed.
        return "nvfp4_ds_mla"

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _indexer_cls(
        self, k_cache: DeepseekV4IndexerCache | None
    ) -> type[DeepseekV4Indexer]:
        return XcpuDeepseekV41Indexer

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = None
        if self.compressor is not None:
            kv_score = self.compressor.fused_wkv_wgate(hidden_states)
        indexer_weights = None
        if self.indexer is not None:
            indexer_weights, _ = self.indexer.weights_proj(hidden_states)
        return qr_kv, kv_score, indexer_weights

    def _opaque_prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        qr_scale: torch.Tensor | None,
        kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> None:
        assert qr_scale is None, "XCPU keeps the q-lora activations in BF16"
        torch.ops.vllm.xcpu_deepseek_v41_prepare_and_attn(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_weights,
            positions,
            attn_out,
            _encode_layer_name(self.prefix),
        )

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        import torch_xcpu

        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = torch_xcpu.ops.deepseek_v41_fused_q_kv_rmsnorm(
            qr, kv, self.q_norm.weight, self.kv_norm.weight, self.eps
        )
        return qr, None, kv

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        import torch_xcpu

        if not isinstance(attn_metadata, dict):
            return q
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        return torch_xcpu.ops.deepseek_v41_qnorm_rope_kv_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.padded_heads,
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        import torch_xcpu

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            output.zero_()
            return
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_tokens = num_decode_tokens + swa_metadata.num_prefill_tokens

        comp_cache = comp_slots = comp_lens = None
        if self.compress_ratio > 0:
            assert self.compressed_cache_prefix is not None
            assert self.topk_indices_buffer is not None
            comp_metadata = attn_metadata[self.compressed_cache_prefix]
            comp_cache = self._compressed_kv_cache()
            comp_slots, comp_lens = (
                torch_xcpu.ops.deepseek_v41_compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_tokens],
                    swa_metadata.token_to_req_indices,
                    comp_metadata.block_table,
                    comp_cache.shape[1],
                    swa_metadata.is_valid_token,
                )
            )

        # Decode and prefill rows keep separate SWA index buffers (both in
        # paged coordinates), so each part runs as its own launch with the
        # step metadata of this layer type and part.
        parts = []
        if num_decode_tokens > 0:
            parts.append((
                "decode",
                0,
                num_decode_tokens,
                swa_metadata.decode_swa_indices.flatten(1),
                swa_metadata.decode_swa_lens,
            ))
        if swa_metadata.num_prefill_tokens > 0:
            parts.append((
                "prefill",
                num_decode_tokens,
                num_tokens,
                swa_metadata.prefill_swa_indices.flatten(1),
                swa_metadata.prefill_swa_lens,
            ))
        for part, start, end, swa_slots, swa_lens in parts:
            torch_xcpu.ops.deepseek_v41_sparse_mla(
                q[start:end],
                self.swa_cache_layer.kv_cache,
                swa_slots,
                swa_lens,
                comp_cache,
                None if comp_slots is None else comp_slots[start:end],
                None if comp_lens is None else comp_lens[start:end],
                self.attn_sink,
                self.scale,
                positions[start:end],
                self.rotary_emb.cos_sin_cache,
                output[start:end],
                swa_metadata.xcpu_sparse_mla[(self.compress_ratio, part)],
            )
        if output.shape[0] > num_tokens:
            output[num_tokens:].zero_()

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        o = attn_out[:, : self.n_local_heads]
        packed = getattr(self.wo_a, "xcpu_wo_a", None)
        if packed is None:
            # The MXFP8 kernel fell back to BF16 emulation: the torch einsum.
            weight = self.wo_a.weight
            wo_a = weight.view(self.n_local_groups, -1, weight.shape[-1])
            z = torch.einsum(
                "tgd,grd->tgr", o.reshape(o.shape[0], self.n_local_groups, -1), wo_a
            ).flatten(1)
        else:
            z = ops.grouped_wo_a(o, *packed)
        return self._wo_b_proj(z)
