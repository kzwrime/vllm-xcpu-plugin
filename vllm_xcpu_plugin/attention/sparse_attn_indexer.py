# SPDX-License-Identifier: Apache-2.0

"""XCPU sparse indexer core and BF16 Q / fused K preprocessing.

The SparseAttnIndexer OOT subclass uses one eager/compile entrypoint.
The metadata builder publishes runtime state under stable handles, and the
entrypoint allocates static scratch tensors before invoking C++, which mirrors upstream
cache insertion, prefill chunks, decode padding, logits and TopK execution.
The upstream Python chain is used only as a test reference in torch_xcpu.
"""

from __future__ import annotations

import functools
import itertools
import os
import weakref

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

# Handles are allocated in model construction order, just like the GDN state.
_metadata_handles = itertools.count(1)


def _publish_runtime_metadata(handles, metadata):
    # UPGRADE: the fields forwarded here are exactly what the C++ mirror of
    # upstream sparse_attn_indexer() reads (torch_xcpu csrc/sparse_indexer.cpp).
    # When DeepseekV32IndexerMetadata or its chunk/decode metadata gains,
    # renames, or changes the meaning of a field, update this function and the
    # C++ side together, or reject the new mode explicitly as done below.
    chunks: list[torch.Tensor] = []
    info: list[int] = []
    decode_tensors: list[torch.Tensor] = []
    if metadata.num_prefills:
        for chunk in metadata.prefill.chunks:
            if chunk.local_cu_seq_lens is None:
                raise ValueError("XCPU indexer requires local cumulative KV lengths")
            chunks.extend((
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                chunk.block_table,
                chunk.local_cu_seq_lens,
            ))
            info.extend((
                chunk.token_start,
                chunk.token_end,
                chunk.max_local_total_seq_lens,
                chunk.local_total_seq_lens,
                int(chunk.skip_kv_gather),
            ))
    padding = False
    if metadata.num_decodes:
        decode = metadata.decode
        if decode.global_seq_lens is not None:
            raise NotImplementedError("XCPU sparse indexer does not support DCP merge")
        decode_tensors = [
            decode.decode_lens,
            decode.seq_lens,
            decode.block_table,
            decode.schedule_metadata,
        ]
        padding = decode.requires_padding
    torch.ops.torch_xcpu.set_sparse_indexer_runtime_metadata(
        handles,
        metadata.num_decode_tokens + metadata.num_prefill_tokens,
        metadata.slot_mapping,
        chunks,
        info,
        metadata.num_decode_tokens,
        decode_tensors,
        padding,
        metadata.seq_lens_cpu,
    )


def _install_runtime_metadata_builder():
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    original_init = DeepseekV32IndexerMetadataBuilder.__init__
    original_build = DeepseekV32IndexerMetadataBuilder.build

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from torch_xcpu.ops_defs.sparse_indexer import (
            get_paged_mqa_logits_metadata_shape,
        )

        from vllm_xcpu_plugin.attention.sparse_mla import validate_sparse_config

        validate_sparse_config(self.vllm_config)
        if self.compress_ratio != 1 or self.indexer_uses_fp4:
            raise NotImplementedError("XCPU indexer requires uncompressed FP8 K cache")
        self.scheduler_metadata_buffer = torch.empty(
            get_paged_mqa_logits_metadata_shape(
                self.vllm_config.scheduler_config.max_num_seqs,
                self.vllm_config.model_config.max_model_len,
                self.num_sms,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.supports_draft_decode_metadata_update = False

    # UPGRADE: wraps DeepseekV32IndexerMetadataBuilder (__init__ and build).
    # Re-check on each vLLM upgrade: new metadata fields, the decode
    # requires_padding rule, the draft-decode update path, and the
    # seq_lens_cpu_upper_bound semantics assumed below.
    @functools.wraps(original_build)
    def build(self, *args, **kwargs):
        metadata = original_build(self, *args, **kwargs)
        # v0.31's CPU lengths are upper bounds used only for kernel selection;
        # device seq_lens remain authoritative for masking and TopK.
        common = kwargs.get("common_attn_metadata")
        if common is None:
            common = args[1]
        metadata.seq_lens_cpu = common.seq_lens_cpu_upper_bound
        if metadata.decode is not None:
            if metadata.decode.indices is not None:
                raise NotImplementedError(
                    "XCPU indexer does not support varlen decode indices"
                )
            torch.ops.torch_xcpu.get_paged_mqa_logits_metadata_out(
                metadata.decode.seq_lens,
                self.kv_cache_spec.num_states,
                metadata.decode.schedule_metadata,
            )
        layers = self.vllm_config.compilation_config.static_forward_context
        handles = [
            layers[name]._xcpu_indexer_metadata_handle
            for name in self.layer_names
            if hasattr(layers[name], "_xcpu_indexer_metadata_handle")
        ]
        if handles:
            _publish_runtime_metadata(handles, metadata)
        return metadata

    DeepseekV32IndexerMetadataBuilder.__init__ = initialize
    DeepseekV32IndexerMetadataBuilder.build = build  # type: ignore[method-assign]


def _sparse_attn_indexer(self, hidden_states, q, k, weights):
    """Shared eager/compile entry; metadata is published before model execution.

    UPGRADE: this and the C++ torch_xcpu::sparse_attn_indexer replace
    vllm.model_executor.layers.sparse_attn_indexer.sparse_attn_indexer(). Any
    new argument, branch or metadata field upstream must be mirrored there or
    rejected here with NotImplementedError. The checklist is on top of the C++
    function; test_indexer_core.py diffs both against the upstream Python.
    """
    if self.use_pcp or self.dcp_world_size != 1 or self.dcp_rank != 0:
        raise NotImplementedError("XCPU sparse indexer core requires PCP=DCP=1")
    if isinstance(q, tuple) or self.use_fp4_cache:
        raise NotImplementedError("XCPU sparse indexer does not support FP4 Q/cache")
    _require_supported_q(q, topk=self.topk_tokens)
    # Sizes depend only on model/scheduler configuration, never runtime metadata.
    # Empty nodes remain outside the op so Inductor can plan/reuse their storage.
    tokens, padded, budget = self._xcpu_workspace_capacity
    rows = (tokens + 7) // 8 * 8
    cols = (self.max_total_seq_len + 511) // 512 * 512
    logits = max(
        min(rows * cols, budget + 7 * cols + rows * 511),
        padded * ((self.max_model_len + 255) // 256 * 256),
    )
    h, d = q.shape[1], self.head_dim
    workspaces = (
        torch.empty(
            (self.max_total_seq_len, d), dtype=torch.float8_e4m3fn, device=q.device
        ),
        torch.empty((self.max_total_seq_len, 4), dtype=torch.uint8, device=q.device),
        torch.empty((logits,), dtype=torch.float32, device=q.device),
        torch.empty((padded * h * d,), dtype=q.dtype, device=q.device),
        torch.empty((padded * h,), dtype=torch.float32, device=q.device),
        torch.empty((padded * self.topk_tokens,), dtype=torch.int32, device=q.device),
    )
    # Profile runs have no live metadata/cache. Reserve the same static scratch
    # capacity, then skip execution. This branch is absent from compiled graphs.
    if (
        not torch.compiler.is_compiling()
        and get_forward_context().attn_metadata is None
    ):
        return
    torch.ops.torch_xcpu.sparse_attn_indexer_from_metadata(
        self.k_cache.kv_cache,
        q,
        k,
        weights,
        self.quant_block_size,
        self.scale_fmt == "ue8m0",
        self.topk_tokens,
        self.max_model_len,
        self.topk_indices_buffer,
        self.skip_k_cache_insert,
        False,
        self.k_cache._xcpu_indexer_metadata_handle,
        *workspaces,
    )


def _require_supported_q(q: torch.Tensor, *, topk: int | None = None) -> None:
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise NotImplementedError(
            "XCPU sparse indexer supports unquantized BF16 Q with head_dim=128 only"
        )
    heads = q.shape[-2]
    if heads not in (32, 64) or (topk is not None and topk not in (512, 1024, 2048)):
        raise NotImplementedError(
            "XCPU sparse indexer supports H={32,64}, D=128 and "
            "K={512,1024,2048}; got "
            f"H={heads}, D={q.shape[-1]}, K={topk}"
        )


def _indexer_k_cache(
    k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    cache: torch.Tensor,
    prefix: str,
    eps: float,
    is_neox: bool,
) -> None:
    forward_context = get_forward_context()
    if forward_context.attn_metadata is None:
        # Capturing this branch would permanently omit the cache write in AOT.
        assert not torch.compiler.is_compiling(), (
            "Indexer profiling without attention metadata must bypass compilation"
        )
        return
    # The uncompressed PCP=DCP=1 indexer uses its own KV-cache group's slots.
    # Keep this Tensor as a graph input, refreshed from the live context each step.
    slot_mappings = forward_context.slot_mapping
    assert isinstance(slot_mappings, dict)
    slots = slot_mappings[prefix]
    import torch_xcpu

    torch_xcpu.ops.fused_indexer_k_norm_rope_cache(
        k,
        weight,
        bias,
        positions,
        cos_sin,
        slots,
        cache,
        eps,
        is_neox,
    )


def _indexer_qk_cache(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    cache: torch.Tensor,
    prefix: str,
    eps: float,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_xcpu

    forward_context = get_forward_context()
    if forward_context.attn_metadata is None:
        # Capturing this branch would permanently omit the cache write in AOT.
        assert not torch.compiler.is_compiling(), (
            "Indexer profiling without attention metadata must bypass compilation"
        )
        return torch_xcpu.ops.fused_indexer_q_rope_quant(
            positions, q, cos_sin, weights, softmax_scale, head_scale, is_neox, False
        )
    # Like the K-only path, refresh this graph input from the live context.
    slot_mappings = forward_context.slot_mapping
    assert isinstance(slot_mappings, dict)
    slots = slot_mappings[prefix]
    return torch_xcpu.ops.fused_indexer_qk_rope_quant_cache(
        positions,
        q,
        cos_sin,
        weights,
        k,
        norm_weight,
        norm_bias,
        slots,
        cache,
        eps,
        softmax_scale,
        head_scale,
        is_neox,
    )


def _indexer_forward(self, hidden_states, qr, positions, rotary_emb):
    import torch_xcpu

    q = self.wq_b(qr)[0].view(-1, self.n_head, self.head_dim)
    kw = self.wk_weights_proj(hidden_states)[0]
    k, weights = kw[:, : self.head_dim], kw[:, self.head_dim :]
    if self._xcpu_fuse_qk_cache:
        q, weights = _indexer_qk_cache(
            q,
            weights,
            k,
            self.k_norm.weight,
            self.k_norm.bias,
            positions,
            rotary_emb.cos_sin_cache,
            self.k_cache.kv_cache,
            self.k_cache.prefix,
            self.k_norm.eps,
            self.softmax_scale,
            self.n_head_scale,
            rotary_emb.is_neox_style,
        )
    else:
        q, weights = torch_xcpu.ops.fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head_scale,
            rotary_emb.is_neox_style,
            False,
        )
        if self._xcpu_fuse_k_cache:
            _indexer_k_cache(
                k,
                self.k_norm.weight,
                self.k_norm.bias,
                positions,
                rotary_emb.cos_sin_cache,
                self.k_cache.kv_cache,
                self.k_cache.prefix,
                self.k_norm.eps,
                rotary_emb.is_neox_style,
            )
        else:
            # Reference path for A/B validation; Q remains BF16 on both paths.
            k = self.k_norm(k)
            k_pe = k[:, : self.rope_dim].unsqueeze(1)
            _, k_pe = rotary_emb(positions, torch.empty_like(k_pe), k_pe)
            k = torch.cat(
                (k_pe.reshape(-1, self.rope_dim), k[:, self.rope_dim :]), dim=-1
            )

    # Let CustomOp dispatch to the registered XCPU OOT implementation.
    return self.indexer_op(hidden_states, q, k, weights)


def _install_indexer_preprocessing() -> None:
    from vllm.model_executor.models.deepseek_v2 import Indexer

    if getattr(Indexer, "_xcpu_preprocessing_installed", False):
        return
    original_init = Indexer.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        parallel = self.vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise NotImplementedError(
                "XCPU DSA indexer preprocessing requires PCP=DCP=1"
            )
        if self.head_dim != 128 or self.rope_dim != 64 or self.n_head not in (32, 64):
            raise NotImplementedError(
                "XCPU DSA indexer requires H={32,64}, D=128, RoPE=64"
            )
        self._xcpu_fuse_k_cache = os.getenv("VLLM_XCPU_FUSED_INDEXER_K", "1") != "0"
        self._xcpu_fuse_qk_cache = (
            self._xcpu_fuse_k_cache
            and os.getenv("VLLM_XCPU_FUSED_INDEXER_QK", "1") != "0"
        )
        self.indexer_op.skip_k_cache_insert = self._xcpu_fuse_k_cache

    Indexer.__init__ = initialize  # type: ignore[method-assign]
    Indexer.forward = _indexer_forward  # type: ignore[method-assign]
    Indexer._xcpu_preprocessing_installed = True


@SparseAttnIndexer.register_oot
class XcpuSparseAttnIndexer(SparseAttnIndexer):
    """XCPU implementation selected through vLLM's CustomOp OOT registry."""

    _upstream_verified = False

    def __init__(self, *args, **kwargs):
        from vllm.config import get_current_vllm_config

        from vllm_xcpu_plugin.upstream_compatibility import (
            verify_upstream_compatibility,
        )

        # Check the common entrypoint, including users outside deepseek_v2.Indexer.
        if not XcpuSparseAttnIndexer._upstream_verified:
            verify_upstream_compatibility(("sparse_indexer",))
            XcpuSparseAttnIndexer._upstream_verified = True
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        spec = config.speculative_config
        next_n = 1 if spec is None else 1 + spec.num_speculative_tokens
        if self.use_pcp or self.dcp_world_size != 1 or self.use_fp4_cache:
            raise NotImplementedError(
                "XCPU sparse indexer requires PCP=DCP=1 and FP8 cache"
            )
        if config.parallel_config.num_ubatches > 1:
            raise NotImplementedError(
                "XCPU sparse indexer runtime state does not support ubatching/DBO"
            )
        if self.quant_block_size != 128 or self.head_dim != 128:
            raise NotImplementedError(
                "XCPU sparse indexer requires D=quant_block_size=128"
            )
        if self.scale_fmt not in ("float32", "ue8m0"):
            raise NotImplementedError(
                "XCPU sparse indexer requires float32 or ue8m0 scales"
            )
        if (
            self.compress_ratio != 1
            or self.candidate_blocks is not None
            or self.candidate_block_size
            or self.candidate_write
        ):
            raise NotImplementedError(
                "XCPU indexer does not support compression or candidate filtering"
            )
        handle = next(_metadata_handles)
        self.k_cache._xcpu_indexer_metadata_handle = handle
        weakref.finalize(
            self.k_cache,
            torch.ops.torch_xcpu.unregister_sparse_indexer_runtime_metadata,
            handle,
        )
        from vllm import envs

        max_tokens = config.scheduler_config.max_num_batched_tokens
        self._xcpu_workspace_capacity = (
            max_tokens,
            min(max_tokens, config.scheduler_config.max_num_seqs) * next_n,
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4,
        )

    def forward_oot(self, hidden_states, q_quant, k, weights):
        return _sparse_attn_indexer(self, hidden_states, q_quant, k, weights)

    # custom_ops=["none"] must still use the same supported XCPU implementation.
    forward_native = forward_oot


# register_oot indexes by class name, but enable/disable uses the original op name.
XcpuSparseAttnIndexer.name = SparseAttnIndexer.name


_integrations_installed = False


def maybe_patch_vllm_sparse_attn_indexer() -> None:
    """Install preprocessing/metadata hooks around the registered OOT CustomOp."""
    global _integrations_installed
    if _integrations_installed:
        return
    _install_runtime_metadata_builder()
    _install_indexer_preprocessing()
    _integrations_installed = True
