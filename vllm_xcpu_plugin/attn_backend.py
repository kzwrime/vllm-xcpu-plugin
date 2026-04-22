import torch
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)


@register_backend(AttentionBackendEnum.TRITON_ATTN)
class XcpuTritonAttentionBackend(TritonAttentionBackend):
    use_direct_unified_op: bool = True
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["XcpuTritonAttentionImpl"]:
        return XcpuTritonAttentionImpl


class XcpuTritonAttentionImpl(TritonAttentionImpl):
    def forward(  # type: ignore[override]
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        import torch_xcpu

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        # max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # seq_threshold_3D = attn_metadata.seq_threshold_3D
        # num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        # softmax_segm_output = attn_metadata.softmax_segm_output
        # softmax_segm_max = attn_metadata.softmax_segm_max
        # softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        # descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        # mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        torch_xcpu.ops.unified_attention(
            q=query,  # query[:num_actual_tokens]
            kv=kv_cache,
            out=output,  # output[:num_actual_tokens]
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=self.scale,
            window_size=1 + self.sliding_window[0],
            block_table=block_table,
            # seq_threshold_3D=seq_threshold_3D,
            # num_par_softmax_segments=num_par_softmax_segments,
            # softmax_segm_output=softmax_segm_output,
            # softmax_segm_max=softmax_segm_max,
            # softmax_segm_expsum=softmax_segm_expsum,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if self.kv_sharing_target_layer_name is not None:
            return

        import torch_xcpu

        torch_xcpu.ops.reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
        )
