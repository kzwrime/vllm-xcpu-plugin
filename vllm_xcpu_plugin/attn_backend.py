from typing import ClassVar

import torch
import torch_xcpu
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.TRITON_MLA)
class XcpuTritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]
    use_direct_unified_op: bool = True

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type["XcpuTritonMLAImpl"]:
        return XcpuTritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadata"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


class XcpuTritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = False

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
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        if sliding_window is None:
            self.sliding_window = (-1, -1)

        unsupported_features = [alibi_slopes, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported"
            )

        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "XcpuTritonMLA V1 with decode context parallel not yet supported"
            )

        if self.q_pad_num_heads is not None:
            raise NotImplementedError(
                "XcpuTritonMLA V1 with q_pad_num_heads not yet supported"
            )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        tmp = torch.bmm(x, self.W_UV)

        # Convert from (N, B, V) to (B, N * V)
        out.copy_(tmp.transpose(0, 1).reshape_as(out))

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # TODO: SDPA
        return

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        # Run MQA
        torch_xcpu.ops.unified_attention(
            q=q,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
            kv=kv_c_and_k_pe_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope]
            out=o,  # [tokens, q_num_heads, kv_lora_rank]
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=self.scale,
            window_size=1 + self.sliding_window[0],
            block_table=block_table,
        )

        return o, None

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLACommonImpl"
            )

        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        # num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        # output = output[:num_actual_toks, ...]
        # q = q[:num_actual_toks, ...]
        # k_c_normed = k_c_normed[:num_actual_toks, ...]
        # k_pe = k_pe[:num_actual_toks, ...]

        # assert (
        #     attn_metadata.num_decodes is not None
        #     and attn_metadata.num_prefills is not None
        #     and attn_metadata.num_decode_tokens is not None
        # )

        # has_decode = attn_metadata.num_decodes > 0
        # has_prefill = attn_metadata.num_prefills > 0
        has_decode = True
        has_prefill = False

        # num_decode_tokens = attn_metadata.num_decode_tokens
        # decode_q = q[:num_decode_tokens]
        # prefill_q = q[num_decode_tokens:]
        # prefill_k_pe = k_pe[num_decode_tokens:]
        # prefill_k_c_normed = k_c_normed[num_decode_tokens:]
        # prefill_output = output[num_decode_tokens:]

        decode_q = q
        prefill_q = q
        prefill_k_pe = k_pe
        prefill_k_c_normed = k_c_normed
        prefill_output = output

        # Write the latent and rope to kv cache
        if kv_cache is not None:
            torch_xcpu.ops.reshape_and_cache(
                k_c_normed,  # [tokens, kv_lora_rank]
                k_pe.squeeze(1),  # [tokens, qk_rope]
                kv_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope]
                attn_metadata.slot_mapping.flatten(),
            )

        if has_prefill:
            self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
                output=prefill_output,
            )

        if has_decode:
            # assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            # Pads the head_dim if necessary (for the underlying kernel)
            N, B, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)

            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            decode_q = (decode_ql_nope, decode_q_pe)

            # call decode attn
            attn_out, lse = self._forward_decode(
                decode_q, kv_cache, attn_metadata, layer
            )

            # v_up projection
            self._v_up_proj(attn_out, out=output)
        return output_padded


@register_backend(AttentionBackendEnum.TRITON_ATTN)
class XcpuTritonAttentionBackend(TritonAttentionBackend):
    use_direct_unified_op: bool = True

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
            return output.fill_(0)

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

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.

            torch_xcpu.ops.reshape_and_cache(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
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
