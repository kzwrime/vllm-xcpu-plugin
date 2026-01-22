import torch
import torch_xcpu
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
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


@SiluAndMul.register_oot
class XcpuSiluAndMul(SiluAndMul):
    """
    自定义 CPU 版本 SiluAndMul 算子
    """

    def forward_cpu(self, input: torch.Tensor) -> torch.Tensor:
        # 输入校验：确保最后一维是 2d 格式
        assert input.dim() >= 1, "Input tensor must have at least 1 dimension"
        assert input.size(-1) % 2 == 0, (
            f"Last dimension of input must be 2*d, got {input.size(-1)}"
        )

        # 计算输出形状：最后一维减半，其余维度不变
        out_shape = list(input.shape)
        out_shape[-1] = out_shape[-1] // 2
        out = torch.empty(out_shape, dtype=input.dtype, device="cpu")

        # 调用 torch_xcpu 底层 C++ 实现
        torch_xcpu.silu_and_mul(out, input)
        return out
