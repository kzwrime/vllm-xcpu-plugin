import math
from typing import ClassVar

import torch
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MultipleOf,
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
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
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
        return attn_metadata


# Used only by the qwen3_dflash context-KV grouped update below.
# The caller validates cache shapes and positive strides before this proof.
def _cache_write_regions_disjoint(caches: list[torch.Tensor]) -> bool:
    """Prove non-overlap from page geometry without reading device slot mappings."""
    regions: list[tuple[int, int, int, int]] = []
    for cache in caches:
        elem = cache.element_size()
        start = cache.data_ptr()
        page_span = elem * (
            1 + sum((n - 1) * s for n, s in zip(cache.shape[1:], cache.stride()[1:]))
        )
        pitch = cache.stride(0) * elem
        end = start + (cache.shape[0] - 1) * pitch + page_span
        for other_start, other_end, other_pitch, other_span in regions:
            if end <= other_start or other_end <= start:
                continue
            # Disjoint residues prove that even interleaved pages cannot overlap.
            period = math.gcd(pitch, other_pitch)
            delta = (start - other_start) % period
            if not (other_span <= delta and page_span <= period - delta):
                return False
        regions.append((start, end, pitch, page_span))
    return True


# Eligibility checks for qwen3_dflash context KV, including inherited DFlash2.
# These safety checks remain active when optional XCPU debug checks are disabled.
def _can_group_cache_updates(
    keys: torch.Tensor,
    values: torch.Tensor,
    caches: list[torch.Tensor],
    slots: list[torch.Tensor],
) -> bool:
    if (
        len(caches) < 2
        or keys.device.type != "mcpu"
        or keys.ndim != 4
        or values.shape != keys.shape
        or keys.dtype not in (torch.bfloat16, torch.float32)
        or values.dtype != keys.dtype
        or values.device != keys.device
        or not keys.is_contiguous()
        or not values.is_contiguous()
        or keys.shape[-1] not in (4, 64, 80, 128, 192, 256)
        or keys.shape[2] == 0
        or keys.data_ptr() % 4
        or values.data_ptr() % 4
    ):
        return False
    source_storage = {
        keys.untyped_storage().data_ptr(),
        values.untyped_storage().data_ptr(),
    }
    source_storage.update(mapping.untyped_storage().data_ptr() for mapping in slots)
    for cache, mapping in zip(caches, slots, strict=True):
        if (
            cache.ndim != 4
            or cache.shape[1] != keys.shape[2]
            or cache.shape[3] != 2 * keys.shape[3]
            or cache.dtype != keys.dtype
            or cache.device != keys.device
            or cache.stride(-1) != 1
            or cache.data_ptr() % 4
            or cache.untyped_storage().data_ptr() in source_storage
            or any(n <= 0 for n in cache.shape)
            or any(s <= 0 for s in cache.stride())
            or any(
                n > 1 and (s * cache.element_size()) % 4
                for n, s in zip(cache.shape[:-1], cache.stride()[:-1])
            )
            or mapping.shape != (keys.shape[1],)
            or mapping.dtype != torch.int64
            or mapping.device != keys.device
            or not mapping.is_contiguous()
        ):
            return False
        span = cache.shape[-1]
        for stride, size in sorted(
            (s, n) for n, s in zip(cache.shape[:-1], cache.stride()[:-1]) if n > 1
        ):
            if stride < span:
                return False
            span += (size - 1) * stride
    return _cache_write_regions_disjoint(caches)


@register_backend(AttentionBackendEnum.TRITON_ATTN)
class XcpuTritonAttentionBackend(TritonAttentionBackend):
    use_direct_unified_op: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    # Performance-tuned XCPU unified attention on real NPU hardware for
    # Qwen3, Qwen3.5, and GLM5.2 requires blocks of at least 64 tokens,
    # aligned to multiples of 64. Expose this to hybrid page alignment too.
    block_size_alignment: ClassVar[int] = 64

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: KVCacheSpec | None = None
    ) -> list[int | MultipleOf]:
        return [MultipleOf(cls.block_size_alignment)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or (
            block_size > 0 and block_size % cls.block_size_alignment == 0
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (4, 64, 80, 128, 192, 256)

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

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
        if cache_dtype_str not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "XCPU attention requires unquantized BF16/FP32 KV"
            )
        if not XcpuTritonAttentionBackend.supports_head_size(head_size):
            raise NotImplementedError(
                f"XCPU attention unsupported head size {head_size}"
            )
        if not XcpuTritonAttentionBackend.supports_block_size(block_size):
            raise ValueError(
                "XCPU attention block size must be a positive multiple of "
                f"{XcpuTritonAttentionBackend.block_size_alignment}."
            )
        kv_cache_shape = TritonAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )
        if kv_cache_shape != (num_blocks, num_kv_heads, block_size, 2 * head_size):
            raise ValueError(f"Unsupported kv cache layout = {kv_cache_shape}")
        return kv_cache_shape

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        stride_order = TritonAttentionBackend.get_kv_cache_stride_order()
        if stride_order not in [(0, 2, 1, 3), (0, 1, 2, 3)]:
            raise ValueError(f"Unsupported kv cache stride order = {stride_order}")
        return TritonAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension
        )

    @staticmethod
    def get_impl_cls() -> type["XcpuTritonAttentionImpl"]:
        return XcpuTritonAttentionImpl


class XcpuTritonAttentionImpl(TritonAttentionImpl):
    # register_attn_backend checks source/signature hashes for the upstream
    # constructor and unified_attention wrapper before installing this backend.
    @staticmethod
    def _check_options(
        alibi_slopes, sinks, logits_soft_cap, use_alibi_sqrt, chunk_lookback
    ) -> None:
        unsupported = {
            "ALiBi": alibi_slopes is not None,
            "attention sinks": sinks is not None,
            "logits soft cap": logits_soft_cap not in (None, 0),
            "ALiBi square root": use_alibi_sqrt,
            "chunked local attention": chunk_lookback != -1,
        }
        for feature, enabled in unsupported.items():
            if enabled:
                raise NotImplementedError(f"XCPU attention does not support {feature}")

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        self._check_options(
            alibi_slopes, sinks, logits_soft_cap, use_alibi_sqrt, chunk_lookback
        )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "XCPU paged attention supports decoder attention only"
            )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "XCPU attention requires unquantized BF16/FP32 KV"
            )
        if not XcpuTritonAttentionBackend.supports_head_size(head_size):
            raise NotImplementedError(
                f"XCPU attention unsupported head size {head_size}"
            )
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError(
                "XCPU attention query heads must be a positive KV-head multiple"
            )
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("XCPU attention sliding window must be positive")
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
            sinks,
            use_alibi_sqrt,
            chunk_lookback,
        )

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
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        import torch_xcpu

        if torch_xcpu.config._ENABLE_CHECKS:
            if output is None:
                raise ValueError("XCPU attention requires an output tensor")
            self._check_options(
                self.alibi_slopes,
                self.sinks,
                self.logits_soft_cap,
                self.use_alibi_sqrt,
                self.chunk_lookback,
            )
            if self.attn_type != AttentionType.DECODER or self.kv_cache_dtype not in (
                "auto",
                "bfloat16",
            ):
                raise NotImplementedError(
                    "XCPU attention requires unquantized decoder KV"
                )
            if (
                query.dtype not in (torch.bfloat16, torch.float32)
                or output.dtype != query.dtype
            ):
                raise NotImplementedError(
                    "XCPU attention requires matching BF16/FP32 query/output"
                )
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(
                    "XCPU attention does not support output quantization scales"
                )

        if attn_metadata is None:
            # Profiling run.
            return output

        if torch_xcpu.config._ENABLE_CHECKS:
            if attn_metadata.use_cascade:
                raise NotImplementedError(
                    "XCPU attention does not support cascade attention"
                )
            for feature in (
                "mm_prefix_range",
                "mm_prefix_range_tensor",
                "rswa_prefix_lens",
                "rswa_window",
            ):
                if getattr(attn_metadata, feature, None) is not None:
                    raise NotImplementedError(
                        f"XCPU attention does not support {feature}"
                    )

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        # max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        causal = attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = True
            raise NotImplementedError("per_seq_causal is not yet supported")

        # seq_threshold_3D = attn_metadata.seq_threshold_3D
        # num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        # softmax_segm_output = attn_metadata.softmax_segm_output
        # softmax_segm_max = attn_metadata.softmax_segm_max
        # softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        # descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        # mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        torch_xcpu.ops.unified_attention(
            q=query,
            kv=kv_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=self.scale,
            window_size=1 + self.sliding_window[0],
            block_table=block_table,
            causal=causal,
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
        import torch_xcpu

        if torch_xcpu.config._ENABLE_CHECKS and (
            self.attn_type != AttentionType.DECODER
            or self.kv_cache_dtype not in ("auto", "bfloat16")
        ):
            raise NotImplementedError(
                "XCPU cache update requires unquantized decoder KV"
            )
        if self.kv_sharing_target_layer_name is not None:
            return

        torch_xcpu.ops.reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
        )

    # Optional hook used only by qwen3_dflash context precompute (also DFlash2).
    # Ordinary attention forwards continue to use do_kv_cache_update per layer.
    def do_kv_cache_update_grouped(
        self,
        layers: list[Attention],
        keys: torch.Tensor,
        values: torch.Tensor,
        slot_mappings: list[torch.Tensor | None],
    ) -> None:
        """Group disjoint context cache writes, otherwise preserve layer order.

        Dynamo tracing uses per-layer updates because the eager safety proof
        inspects storage pointers. Context precompute normally runs outside the
        compiled model forward, so compile services can still use this hook.
        """
        import torch_xcpu

        active = [
            (i, layer, slots)
            for i, (layer, slots) in enumerate(zip(layers, slot_mappings, strict=True))
            if slots is not None
        ]
        compatible = (
            not torch.compiler.is_compiling()
            and keys.ndim == 4
            and keys.shape[0] == len(layers)
        ) and all(
            type(layer.impl) is XcpuTritonAttentionImpl
            and "do_kv_cache_update" not in vars(layer.impl)
            and layer.impl.attn_type == AttentionType.DECODER
            and layer.impl.kv_cache_dtype in ("auto", "bfloat16")
            for _, layer, _ in active
        )
        if compatible:
            active = [
                (i, layer, slots)
                for i, layer, slots in active
                if layer.impl.kv_sharing_target_layer_name is None
            ]
            if not active:
                return
            caches = [layer.kv_cache for _, layer, _ in active]
            slots = [mapping for _, _, mapping in active]
            if _can_group_cache_updates(keys, values, caches, slots):
                torch_xcpu.ops.reshape_and_cache_grouped(
                    keys, values, caches, slots, [i for i, _, _ in active]
                )
                return

        for i, layer, slots in active:
            layer.impl.do_kv_cache_update(
                layer, keys[i], values[i], layer.kv_cache, slots
            )
