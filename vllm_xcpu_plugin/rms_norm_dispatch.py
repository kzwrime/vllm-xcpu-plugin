"""MCPU dispatch for vLLM's shared and grouped RMSNorm custom operation."""

import torch

_LIBRARY: torch.library.Library | None = None


def _rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    import torch_xcpu.ops as ops

    # These checks are unconditional: native debug checks may be disabled.
    if input.ndim not in (2, 3, 4) or input.shape[-1] == 0:
        raise ValueError("rms_norm requires rank 2/3/4 and a nonempty hidden dimension")
    if input.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("rms_norm requires float32, bfloat16 or float16 input")
    if out.shape != input.shape or out.dtype != input.dtype:
        raise ValueError("rms_norm output shape/dtype must match input")
    if out.device != input.device:
        raise ValueError("rms_norm output device must match input")
    if not out.is_contiguous():
        raise ValueError("rms_norm requires contiguous output")
    if weight is not None:
        if weight.dtype != input.dtype or weight.device != input.device:
            raise ValueError("rms_norm weight dtype/device must match input")
        if weight.ndim not in (1, 2) or weight.shape[-1] != input.shape[-1]:
            raise ValueError("rms_norm weight must have shape [H] or [L, H]")
        if not weight.is_contiguous():
            raise ValueError("rms_norm requires contiguous weight")
        if weight.ndim == 2 and (input.ndim < 2 or weight.shape[0] != input.shape[0]):
            raise ValueError(
                "rms_norm grouped weight must match input's first dimension"
            )
    if input.numel() == 0:
        return

    # Like upstream, pack a strided hidden dimension without losing layer axes.
    x = input
    if x.stride(-1) != 1:
        x = x.contiguous()
    # Native DMA requires four-byte aligned rows/transfers. Host Release builds
    # tolerate unaligned BF16 memcpy, but debug builds and NPU DMA need this too.
    unaligned_bf16 = input.dtype == torch.bfloat16 and (
        x.shape[-1] % 2 != 0
        or x.data_ptr() % 4 != 0
        or out.data_ptr() % 4 != 0
        or any(
            size > 1 and stride % 2 != 0
            for size, stride in zip(x.shape[:-1], x.stride()[:-1])
        )
        or (weight is not None and weight.data_ptr() % 4 != 0)
    )
    # The native kernels accumulate in FP32. Packing FP16 and unaligned BF16
    # into FP32 preserves their values and rounds only when copying the output.
    if input.dtype == torch.float16 or unaligned_bf16:
        x = x.float()
    if weight is None:
        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    else:
        w = weight.to(dtype=x.dtype).contiguous()

    direct = (
        out.is_contiguous()
        and out.dtype == x.dtype
        and not torch._C._overlaps(out, input)
        and (weight is None or not torch._C._overlaps(out, weight))
    )
    result = (
        out.view(x.shape)
        if direct
        else torch.empty(x.shape, device=x.device, dtype=x.dtype)
    )
    ops.rms_norm(result, x, w, epsilon)
    if not direct:
        out.copy_(result.reshape(input.shape))


def register_rms_norm_kernel() -> None:
    """Keep the upstream schema and route PrivateUse1 to native grouped RMS."""
    global _LIBRARY
    if _LIBRARY is not None:
        return
    # Import loads the upstream _C schema before registering its implementation.
    import vllm._custom_ops  # noqa: F401

    library = torch.library.Library("_C", "IMPL", "PrivateUse1")
    library.impl("rms_norm", _rms_norm)
    _LIBRARY = library
