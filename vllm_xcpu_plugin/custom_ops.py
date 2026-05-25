import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)


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

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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


@SiluAndMul.register_oot
class XcpuSiluAndMul(SiluAndMul):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_oot(x: torch.Tensor) -> torch.Tensor:
        import torch_xcpu

        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch_xcpu.ops.silu_and_mul(out, x)
        return out


@VocabParallelEmbedding.register_oot
class XcpuVocabParallelEmbedding(VocabParallelEmbedding):
    """XCPU implementation of VocabParallelEmbedding using torch_xcpu ops."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = 64,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )

    def forward_oot(self, input_):

        if self.tp_size > 1:
            import torch_xcpu.ops as ops

            masked_input = torch.empty_like(input_)
            input_mask = torch.empty(
                input_.shape, dtype=torch.bool, device=input_.device
            )
            ops.get_masked_input_and_mask(
                masked_input,
                input_mask,
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_

        # Get the embeddings.
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            import torch_xcpu.ops as ops

            assert isinstance(self.weight, torch.Tensor)
            if self.tp_size > 1:
                output_parallel = ops.fused_embedding_masked_fill(
                    self.weight, masked_input, input_mask
                )
            else:
                output_parallel = ops.embedding(self.weight, masked_input)
            # output_parallel = torch.nn.functional.embedding(
            #     masked_input.long(), self.weight
            # )
        else:
            output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1 and not isinstance(self.quant_method,
                                               UnquantizedEmbeddingMethod):
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        from vllm.distributed import tensor_model_parallel_all_reduce

        output = tensor_model_parallel_all_reduce(output_parallel)
        return output
