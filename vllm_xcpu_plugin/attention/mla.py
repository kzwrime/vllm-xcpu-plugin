# SPDX-License-Identifier: Apache-2.0
"""Sparse-only MLA projections behind the v0.31 pluggable wrapper."""

import os

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.utils import replace_parameter

from vllm_xcpu_plugin import envs

logger = init_logger(__name__)


class XcpuMLAAttention(MLAAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fuse_rope_cache = os.getenv("VLLM_XCPU_FUSED_MLA", "1") != "0"
        # Fuse FP8 sparse attention with the following V up-projection; "0"
        # keeps the separate forward_mqa + einsum path for debugging.
        self._fuse_vproj = os.getenv("VLLM_XCPU_FUSED_MLA_VPROJ", "1") != "0"
        # Callbacks of the owning wrapper that need W_UK_T, run at the end of
        # process_weights_after_loading (after all linear layers are processed).
        self._post_load_hooks: list = []

    def _forward_mqa_vproj(self, q, kv_cache, metadata, output):
        # output: [tokens, heads, v_head_dim] rows.
        if self._fuse_vproj and self.impl.kv_cache_dtype == "fp8_ds_mla":
            self.impl.forward_mqa_vproj(q, kv_cache, metadata, self, output)
            return
        import torch_xcpu

        latent, _ = self.impl.forward_mqa(q, kv_cache, metadata, self)
        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(latent, self.W_UV, output)

    def _fused_rope_inputs(self, scaling):
        """(metadata, slots) when the fused RoPE/cache path applies, else None."""
        context = get_forward_context()
        metadata = context.attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata[self.layer_name]
        if metadata is None:
            assert not torch.compiler.is_compiling(), (
                "MLA profiling must bypass compilation"
            )
        slots = context.slot_mapping.get(self.layer_name)
        if (
            not self._fuse_rope_cache
            or self.use_pcp
            or metadata is None
            or slots is None
            or self.kv_cache.numel() == 0
            or scaling is not None
        ):
            return None
        return metadata, slots

    def _xcpu_kv_cache_dtype(self):
        return "auto" if self.kv_cache_dtype == "bfloat16" else self.kv_cache_dtype

    def _attend_fused(
        self, q, kv_c, k_pe, positions, rotary_emb, metadata, slots, output_shape
    ):
        """Fused RoPE + cache insertion + attention + V up-projection.

        ``q`` is the raw projection ``[T, H, 256]``; the q up-projection, the
        Q_pe / K_pe RoPE and the cache insertion run as one torch_xcpu op.
        ``metadata`` / ``slots`` come from :meth:`_fused_rope_inputs`.
        """
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
            self._xcpu_kv_cache_dtype(),
            rotary_emb.is_neox_style,
        )
        return self._attend_projected(projected, metadata, output_shape)

    def _attend_projected(self, projected, metadata, output_shape):
        """Attention + V up-projection on a ready query ``[T, H, 576]``."""
        tokens = projected.shape[0]
        output = projected.new_empty(output_shape)
        self._forward_mqa_vproj(
            projected,
            self.kv_cache,
            metadata,
            output[:tokens].view(tokens, self.num_heads, self.v_head_dim),
        )
        return output

    def forward_with_rope(self, q, kv_c, k_pe, positions, rotary_emb, scaling):
        fused = self._fused_rope_inputs(scaling)
        output_shape = (q.shape[0], self.num_heads * self.v_head_dim)
        if fused is None:
            q[..., self.qk_nope_head_dim :], k_pe = rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
            if scaling is not None:
                q = q * scaling
            return self(q, kv_c, k_pe, output_shape=output_shape)
        metadata, slots = fused
        return self._attend_fused(
            q, kv_c, k_pe, positions, rotary_emb, metadata, slots, output_shape
        )

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
        for hook in self._post_load_hooks:
            hook()

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
        self._forward_mqa_vproj(
            (projected, q_pe),
            kv_cache,
            attn_metadata,
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
        # GLM-5.2 TP4 (16 heads/rank) can fuse q/kv norm, q_b_proj, W_UK_T, RoPE and
        # the KV-cache write into one torch_xcpu op (VLLM_XCPU_FUSED_MLA_QNORM_QB_UK,
        # default off). The op reads q_b_proj's FP8 weights as packed by the linear
        # backend; _enable_qnorm_qb_uk decides once weights are loaded.
        self._use_qnorm_qb_uk = False
        if (
            envs.VLLM_XCPU_FUSED_MLA_QNORM_QB_UK is not None
            and self.num_heads == 16
            and self.qk_nope_head_dim == 192
            and self.kv_lora_rank == 512
            and self.q_lora_rank == 2048
            and self.qk_rope_head_dim == 64
        ):
            self.mla_attn._post_load_hooks.append(self._enable_qnorm_qb_uk)
        elif envs.VLLM_XCPU_FUSED_MLA_QNORM_QB_UK is not None:
            logger.warning_once(
                "fused_mla_qnorm_qb_uk requested (version=%s) but disabled: it needs "
                "16 heads per rank (got %d; GLM-5.2 needs TP4) with nope=192, "
                "rope=64, kv_lora=512, q_lora=2048",
                envs.VLLM_XCPU_FUSED_MLA_QNORM_QB_UK,
                self.num_heads,
                scope="process",
            )
        if not type(self)._upstream_verified:
            from vllm_xcpu_plugin.upstream_compatibility import (
                verify_upstream_compatibility,
            )

            verify_upstream_compatibility(("sparse_mla",))
            type(self)._upstream_verified = True

    def _enable_qnorm_qb_uk(self):
        """After weight loading: use the fused op if the packed q_b_proj weights fit."""
        import torch_xcpu

        fp8_linear = getattr(self.q_b_proj, "_xcpu_fp8_linear", None)
        self._use_qnorm_qb_uk = (
            fp8_linear is not None
            and torch_xcpu.ops.fused_mla_qnorm_qb_uk_rope_cache_supported(
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                fp8_linear.params,
                self.mla_attn.W_UK_T,
            )
        )
        backend = fp8_linear.params.backend.name.lower() if fp8_linear else None
        if self._use_qnorm_qb_uk:
            # Plugin loggers have no INFO handler in MP workers; warning_once is the
            # level the FP8 linear backend line uses too.
            logger.warning_once(
                "fused_mla_qnorm_qb_uk enabled: version=%s fp8_backend=%s",
                envs.VLLM_XCPU_FUSED_MLA_QNORM_QB_UK,
                backend,
                scope="process",
            )
        else:
            logger.warning_once(
                "fused_mla_qnorm_qb_uk requested (version=%s) but disabled: "
                "fp8_backend=%s is not supported by the fused op",
                envs.VLLM_XCPU_FUSED_MLA_QNORM_QB_UK,
                backend,
                scope="process",
            )

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        if self.q_lora_rank is None or self.dcp_q_replicate or self.rotary_emb is None:
            raise NotImplementedError(
                "XCPU sparse MLA requires q-LoRA and RoPE without DCP"
            )
        qkv = self.fused_qkv_a_proj(hidden_states)[0]
        if self._use_qnorm_qb_uk:
            rope_inputs = self.mla_attn._fused_rope_inputs(llama_4_scaling)
            if rope_inputs is not None:
                return self._forward_qnorm_qb_uk(
                    positions, hidden_states, qkv, rope_inputs
                )
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
        return self._project_output(hidden_states, output)

    def _project_output(self, hidden_states, output):
        if self.g_proj is not None:
            output = output * self.g_proj(hidden_states)[0].sigmoid()
        return self.o_proj(output)[0]

    def _forward_qnorm_qb_uk(self, positions, hidden_states, qkv, rope_inputs):
        """q_a/kv_a norm + q_b_proj + W_UK_T + RoPE + cache write as one torch_xcpu op.

        Runs on all ``qkv`` rows (as the unfused path does); rows beyond the slot
        mapping keep Q_pe unroped and write no cache entry.
        """
        import torch_xcpu

        mla_attn = self.mla_attn
        metadata, slots = rope_inputs
        mla_attn.prepare_kv_cache_update(metadata)
        q_c, _kv_c, out_q = torch_xcpu.ops.fused_mla_qnorm_qb_uk_rope_cache(
            qkv,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            self.kv_a_layernorm.variance_epsilon,
            self.q_b_proj._xcpu_fp8_linear.params,
            mla_attn.W_UK_T,
            positions,
            self.rotary_emb.cos_sin_cache,
            slots,
            mla_attn.kv_cache,
            mla_attn._xcpu_kv_cache_dtype(),
            self.rotary_emb.is_neox_style,
        )
        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        mla_attn.impl.record_logical_topk_ready()
        tokens = (
            qkv.shape[0]
            if torch.compiler.is_compiling()
            else metadata.num_actual_tokens
        )
        output = mla_attn._attend_projected(
            out_q[:tokens], metadata, (qkv.shape[0], self.num_heads * self.v_head_dim)
        )
        return self._project_output(hidden_states, output)
