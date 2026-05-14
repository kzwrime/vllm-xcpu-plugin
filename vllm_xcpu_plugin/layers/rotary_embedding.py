import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


@RotaryEmbedding.register_oot
class XcpuRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._match_cos_sin_cache_dtype(query)

        import torch_xcpu.ops as ops

        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key


@Llama3RotaryEmbedding.register_oot
class XcpuLlama3RotaryEmbedding(Llama3RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            scaling_factor,
            low_freq_factor,
            high_freq_factor,
            orig_max_position,
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._match_cos_sin_cache_dtype(query)

        import torch_xcpu.ops as ops

        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key


@MRotaryEmbedding.register_oot
class XcpuMRotaryEmbedding(MRotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is None:
            raise NotImplementedError("torch_xcpu MRotaryEmbedding requires key")
        if offsets is not None:
            raise NotImplementedError(
                "torch_xcpu MRotaryEmbedding does not support offsets"
            )
        if query.dim() != 2 or key.dim() != 2:
            raise NotImplementedError(
                "torch_xcpu MRotaryEmbedding only supports 2D query/key"
            )
        if not query.is_contiguous() or not key.is_contiguous():
            raise NotImplementedError(
                "torch_xcpu MRotaryEmbedding only supports contiguous query/key"
            )

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        import torch_xcpu.ops as ops

        ops.mrope(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.mrope_section,
            self.is_neox_style,
            self.mrope_interleaved,
        )
        return query, key
