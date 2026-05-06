import torch
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


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
