# SPDX-License-Identifier: Apache-2.0
"""Sparse-only MLA projections behind the v0.31 pluggable wrapper."""

import os

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.utils import replace_parameter

from vllm_xcpu_plugin import shared_kv_cache


class XcpuMLAAttention(MLAAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fuse_rope_cache = os.getenv("VLLM_XCPU_FUSED_MLA", "1") != "0"

    def forward_with_rope(self, q, kv_c, k_pe, positions, rotary_emb, scaling):
        context = get_forward_context()
        metadata = context.attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata[self.layer_name]
        if metadata is None:
            assert not torch.compiler.is_compiling(), (
                "MLA profiling must bypass compilation"
            )
        slots = context.slot_mapping.get(self.layer_name)
        output_shape = (q.shape[0], self.num_heads * self.v_head_dim)
        if (
            not self._fuse_rope_cache
            or self.use_pcp
            or shared_kv_cache.enabled()
            or metadata is None
            or slots is None
            or self.kv_cache.numel() == 0
            or scaling is not None
        ):
            q[..., self.qk_nope_head_dim :], k_pe = rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
            if scaling is not None:
                q = q * scaling
            return self(q, kv_c, k_pe, output_shape=output_shape)
        import torch_xcpu

        # This backend rejects HiSparse/CP at construction. Still enter the
        # v0.31 preparation lifecycle before writing this layer's cache.
        self.prepare_kv_cache_update(metadata)
        tokens = (
            q.shape[0] if torch.compiler.is_compiling() else metadata.num_actual_tokens
        )
        projected = q.new_empty((
            tokens,
            self.num_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
        ))
        torch_xcpu.ops.fused_mla_rope_qproj_cat_cache(
            q[:tokens],
            self.W_UK_T,
            k_pe[:tokens].squeeze(1),
            kv_c[:tokens],
            positions[:tokens],
            rotary_emb.cos_sin_cache,
            slots,
            self.kv_cache,
            projected,
            "auto" if self.kv_cache_dtype == "bfloat16" else self.kv_cache_dtype,
            rotary_emb.is_neox_style,
        )
        latent, _ = self.impl.forward_mqa(projected, self.kv_cache, metadata, self)
        output = q.new_empty(output_shape)
        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(
            latent,
            self.W_UV,
            output[:tokens].view(tokens, self.num_heads, self.v_head_dim),
        )
        return output

    def process_weights_after_loading(self, act_dtype):
        self.impl.process_weights_after_loading(act_dtype)
        weight = self.kv_b_proj.weight
        if weight.dtype != act_dtype or weight.shape != (
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            self.kv_lora_rank,
        ):
            raise ValueError(
                "XCPU MLA requires canonical dequantized kv_b_proj weights"
            )
        weight = weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        key, value = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        # Own both projections before releasing the canonical source. Merely
        # calling contiguous() can retain its storage for e.g. a single head.
        key = key.clone(memory_format=torch.contiguous_format)
        value = value.transpose(1, 2).clone(memory_format=torch.contiguous_format)
        replace_parameter(self, "W_UK_T", key)
        replace_parameter(self, "W_UV", value)

        # Sparse prefill and decode only use the projections above. Keep empty
        # parameters on the original device, as in native AMX MLA, but release
        # the dequantized intermediate and its now-unused checkpoint scales.
        for name in ("weight", "weight_scale", "weight_scale_inv"):
            parameter = getattr(self.kv_b_proj, name, None)
            if isinstance(parameter, torch.Tensor):
                replace_parameter(self.kv_b_proj, name, parameter.new_empty(0))

    def forward_impl(
        self,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
        quant_group_size=None,
        quant_scale_ue8m0=None,
        quant_col_major=None,
        quant_tma_aligned=None,
        q_dcp_replicated=None,
    ):
        if (
            output_scale is not None
            or output_block_scale is not None
            or output.dtype != q.dtype
            or q_dcp_replicated is not None
        ):
            raise NotImplementedError(
                "XCPU sparse MLA requires unquantized output and local Q"
            )
        if attn_metadata is None:
            return output.zero_()
        import torch_xcpu

        # MCPU does not capture padded CUDA graphs. Keep the token dimension
        # symbolic under compile instead of specializing on Python metadata.
        tokens = (
            q.shape[0]
            if torch.compiler.is_compiling()
            else attn_metadata.num_actual_tokens
        )
        q = q[:tokens]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        projected = q.new_empty((tokens, self.num_heads, self.kv_lora_rank))
        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(q_nope, self.W_UK_T, projected)
        latent, _ = self.impl.forward_mqa(
            (projected, q_pe), kv_cache, attn_metadata, self
        )
        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(
            latent,
            self.W_UV,
            output[:tokens].view(tokens, self.num_heads, self.v_head_dim),
        )
        return output


@MultiHeadLatentAttentionWrapper.register_oot
class XcpuMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    attention_cls = XcpuMLAAttention
    _upstream_verified = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_sparse:
            raise NotImplementedError("MCPU supports sparse MLA (DSA) only")
        if not type(self)._upstream_verified:
            from vllm_xcpu_plugin.upstream_compatibility import (
                verify_upstream_compatibility,
            )

            verify_upstream_compatibility(("sparse_mla",))
            type(self)._upstream_verified = True

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        if self.q_lora_rank is None or self.dcp_q_replicate or self.rotary_emb is None:
            raise NotImplementedError(
                "XCPU sparse MLA requires q-LoRA and RoPE without DCP"
            )
        qkv = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], -1
        )
        kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        if self.fuse_qkv_rmsnorm:
            from vllm.models.common.ops import fused_q_kv_rmsnorm

            q_c, kv_c = fused_q_kv_rmsnorm(
                q_c,
                kv_c,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                self.q_a_layernorm.variance_epsilon,
            )
        else:
            q_c = self.q_a_layernorm(q_c)
            kv_c = self.kv_a_layernorm(kv_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        self.mla_attn.impl.record_logical_topk_ready()
        output = self.mla_attn.forward_with_rope(
            q, kv_c, k_pe.unsqueeze(1), positions, self.rotary_emb, llama_4_scaling
        )
        if self.g_proj is not None:
            output = output * self.g_proj(hidden_states)[0].sigmoid()
        return self.o_proj(output)[0]
