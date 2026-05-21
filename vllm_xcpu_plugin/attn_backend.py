import torch
import torch.nn as nn
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.mla_attention import MLACommonBackend
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
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
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)


class XcpuTritonAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        attn_metadata.kv_cache_tensor_layout = "KV_BLOCK"
        return attn_metadata


@register_backend(AttentionBackendEnum.TRITON_MLA)
class XcpuTritonMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls():
        return XcpuTritonMLAAttention

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadata"]:  # type: ignore[override]
        return TritonAttentionMetadataBuilder  # type: ignore[return-value]

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


class XcpuTritonMLAAttention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        self.indexer = indexer

        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False
        self.quant_config = quant_config

        # Initialize KV cache quantization attributes
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales

        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

        self.use_sparse = use_sparse

        vllm_config = get_current_vllm_config_or_none()
        self.dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Attributes for forward_impl method
        self._vllm_config = get_current_vllm_config()
        self._chunked_prefill_workspace_size: int | None = None

        self.sliding_window = (-1, -1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported"
            )

        dtype = torch.get_default_dtype()
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            kv_cache_dtype,
            use_mla=True,
            use_sparse=use_sparse,
            num_heads=self.num_heads,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        assert self.use_direct_call
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]  # type: ignore[assignment]
            self_kv_cache = self.kv_cache
            slot_mapping = forward_context.slot_mapping

            assert isinstance(slot_mapping, dict), (
                f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            )
            assert self.attn_backend.accept_output_buffer
            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)  # type: ignore[arg-type]
                self.forward_impl(
                    q,
                    kv_c_normed,
                    k_pe,
                    self_kv_cache,
                    attn_metadata,  # type: ignore[arg-type]
                    output=output,
                )
                return output

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T

        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # # Convert from (B, N, L) to (N, B, L)
        # x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        # tmp = torch.bmm(x, self.W_UV)
        # # Convert from (N, B, V) to (B, N * V)
        # out.copy_(tmp.transpose(0, 1).reshape_as(out))

        # x = x.view(-1, self.num_heads, self.kv_lora_rank)
        # out = out.view(-1, self.num_heads, self.v_head_dim)
        import torch_xcpu

        torch_xcpu.ops.einsum_mhk_hkn_to_mhn(x, self.W_UV, out)
        # out = x[..., :self.v_head_dim]
        # out = out.view(-1, self.num_heads * self.v_head_dim)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,  # type: ignore[override]
        output: torch.Tensor,
    ) -> None:
        # TODO: SDPA
        return

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,  # type: ignore[override]
        output: torch.Tensor,
    ) -> torch.Tensor:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        import torch_xcpu  # noqa: E402

        torch_xcpu.ops.unified_attention(
            q=q,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
            kv=kv_c_and_k_pe_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope]
            out=output,  # [tokens, q_num_heads, kv_lora_rank]
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=self.scale,
            window_size=1 + self.sliding_window[0],
            block_table=block_table,
        )
        return output

    def forward_impl(
        self,
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
            return output

        # num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        # output_padded = output
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
        import torch_xcpu  # noqa: E402

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
                output=prefill_output,
            )

        if has_decode:
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            B, N, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape

            # # Convert from (B, N, P) to (N, B, P)
            # decode_q_nope = decode_q_nope.transpose(0, 1)
            # # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            # decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            # # Convert from (N, B, L) to (B, N, L)
            # decode_ql_nope = decode_ql_nope.transpose(0, 1)

            decode_ql_nope = decode_q_nope.new_empty((B, N, L))

            import torch_xcpu

            torch_xcpu.ops.einsum_mhk_hkn_to_mhn(
                decode_q_nope, self.W_UK_T, decode_ql_nope
            )

            decode_q = torch.cat((decode_ql_nope, decode_q_pe), dim=-1)

            attn_out = self._forward_decode(
                decode_q, kv_cache, attn_metadata, decode_ql_nope
            )

            self._v_up_proj(attn_out, out=output)
        return output


MLAAttention.register(XcpuTritonMLAAttention)


@register_backend(AttentionBackendEnum.TRITON_ATTN)
class XcpuTritonAttentionBackend(TritonAttentionBackend):
    use_direct_unified_op: bool = True
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_builder_cls() -> type["XcpuTritonAttentionMetadataBuilder"]:
        return XcpuTritonAttentionMetadataBuilder

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
