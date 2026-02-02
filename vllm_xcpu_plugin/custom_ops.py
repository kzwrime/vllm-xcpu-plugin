import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    import torch_xcpu.ops as ops

    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_xcpu.ops as ops

    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


@RMSNorm.register_oot
class XcpuRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        # logger.info("Init XcpuRMSNorm")

        if current_platform.is_cpu():
            self._forward_method = self.forward_cpu

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # logger.debug("XcpuRMSNorm.forward_cpu")

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            assert residual is not None
            assert self.weight.data is not None
            return fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        else:
            return rms_norm(x, self.weight.data, self.variance_epsilon)


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

        if current_platform.is_cpu():
            self._forward_method = self.forward_cpu

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._match_cos_sin_cache_dtype(query)

        import torch_xcpu.ops as ops

        # rotary_embedding() is an in-place operation
        # that updates the query and key tensors.
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key
@SiluAndMul.register_oot
class XcpuSiluAndMul(SiluAndMul):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:silu_and_mul]

    def __init__(self):
        super().__init__()
        if current_platform.is_cpu():
            self._forward_method = self.forward_cpu
            import torch_xcpu

            self.op = torch_xcpu.ops.silu_and_mul

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out
