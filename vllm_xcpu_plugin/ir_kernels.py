import torch
from torch import Tensor
from vllm import ir


def _supports_rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    del epsilon
    return (
        variance_size is None
        and weight is not None
        and 2 <= x.dim() <= 4
        and x.stride(-1) == 1
        and weight.dim() in (1, 2)
        and weight.shape[-1] == x.shape[-1]
        and (weight.dim() == 1 or weight.shape[0] == x.shape[0])
        and x.dtype in (torch.bfloat16, torch.float32)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
    )


@ir.ops.rms_norm.register_impl(
    "torch_xcpu",
    supports_args=_supports_rms_norm,
)
def rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    assert weight is not None
    assert variance_size is None

    import torch_xcpu

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    torch_xcpu.ops.rms_norm(out, x, weight, epsilon)
    return out


def _supports_fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    del epsilon
    return (
        variance_size is None
        and weight is not None
        and x.dim() <= 2
        and x.shape == x_residual.shape
        and x.dtype in (torch.bfloat16, torch.float32)
        and x_residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x_residual.device == x.device
        and weight.device == x.device
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and weight.is_contiguous()
    )


def _supports_fused_add_rms_norm_inplace(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return _supports_fused_add_rms_norm(x, x_residual, weight, epsilon, variance_size)


@ir.ops.fused_add_rms_norm.register_impl(
    "torch_xcpu_inplace",
    supports_args=_supports_fused_add_rms_norm_inplace,
    inplace=True,
)
def fused_add_rms_norm_inplace(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert weight is not None
    assert variance_size is None

    import torch_xcpu

    torch_xcpu.ops.fused_add_rms_norm(x, x_residual, weight, epsilon)
    return x, x_residual


@ir.ops.fused_add_rms_norm.register_impl(
    "torch_xcpu",
    supports_args=_supports_fused_add_rms_norm,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert weight is not None
    assert variance_size is None

    import torch_xcpu

    out = torch.empty_like(x)
    residual_out = torch.empty_like(x_residual)
    torch_xcpu.ops.fused_add_rms_norm_out(
        out, residual_out, x, x_residual, weight, epsilon
    )
    return out, residual_out
