import torch
from torch import Tensor
from vllm.logger import logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins


def _is_torch_xcpu_op_available(op_name: str) -> bool:
    try:
        return hasattr(torch.ops.torch_xcpu, op_name)
    except Exception:
        return False


if hasattr(SiluAndMul, "op_registry_oot"):
    SiluAndMul.op_registry_oot = {}
    logger.info("清空原生SiluAndMul的op_registry_oot，避免重复注册")


def _register_silu_and_mul_op():
    if hasattr(torch.ops.torch_xcpu, "silu_and_mul"):
        logger.info("silu_and_mul already registered, skip")
        return

    try:
        torch.library.define(
            "torch_xcpu::silu_and_mul", "(Tensor(a!) out, Tensor input) -> Tensor(a!)"
        )

        @torch.library.impl("torch_xcpu::silu_and_mul", "CPU")
        def _silu_and_mul_impl(out: Tensor, input: Tensor):
            try:
                d = input.shape[-1] // 2
                a = input[..., :d]
                b = input[..., d:]

                if input.dtype == torch.float32 and _is_torch_xcpu_op_available(
                    "silu_and_mul_fp32"
                ):
                    torch.ops.torch_xcpu.silu_and_mul_fp32(out, a, b)
                elif input.dtype == torch.bfloat16 and _is_torch_xcpu_op_available(
                    "silu_and_mul_bf16"
                ):
                    torch.ops.torch_xcpu.silu_and_mul_bf16(out, a, b)
                else:
                    out.copy_(torch.nn.functional.silu(a) * b)
            except Exception as e:
                logger.warning(f"silu_and_mul impl failed, fallback to native: {e}")
                d = input.shape[-1] // 2
                a = input[..., :d]
                b = input[..., d:]
                out.copy_(torch.nn.functional.silu(a) * b)
            return out

        if hasattr(torch.ops.torch_xcpu, "silu_and_mul"):
            logger.info("silu_and_mul registered to torch.ops.torch_xcpu successfully")
    except Exception as e:
        logger.error(f"Failed to register silu_and_mul: {e}")

        @torch.library.impl("torch_xcpu::silu_and_mul", "CPU")
        def _silu_and_mul_fallback(out: Tensor, input: Tensor):
            d = input.shape[-1] // 2
            a = input[..., :d]
            b = input[..., d:]
            out.copy_(torch.nn.functional.silu(a) * b)
            return out


@SiluAndMul.register_oot
class XcpuSiluAndMul(SiluAndMul):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Init XcpuSiluAndMul (XCPU backend)")

        if current_platform.is_cpu():
            try:
                self._forward_method = self.forward_cpu
                fp32_ok = _is_torch_xcpu_op_available("silu_and_mul_fp32")
                bf16_ok = _is_torch_xcpu_op_available("silu_and_mul_bf16")
                logger.info(f"XCPU算子可用性：fp32={fp32_ok}, bf16={bf16_ok}")
            except Exception as e:
                logger.warning(f"Failed to set forward_cpu, fallback to native: {e}")
                self._forward_method = self.forward_native
        else:
            self._forward_method = self.forward_native

    def forward_cpu(self, input: Tensor) -> Tensor:
        if input.size(-1) % 2 != 0:
            raise ValueError(
                f"SiluAndMul input last dim must be even, got {input.size(-1)}"
            )

        d_out = input.shape[-1] // 2
        out = torch.empty(
            *input.shape[:-1], d_out, dtype=input.dtype, device=input.device
        )

        if input.dtype == torch.float32 and _is_torch_xcpu_op_available(
            "silu_and_mul_fp32"
        ):
            d = input.shape[-1] // 2
            a = input[..., :d].contiguous()
            b = input[..., d:].contiguous()
            out = out.contiguous()
            torch.ops.torch_xcpu.silu_and_mul_fp32(out, a, b)
        elif input.dtype == torch.bfloat16 and _is_torch_xcpu_op_available(
            "silu_and_mul_bf16"
        ):
            d = input.shape[-1] // 2
            a = input[..., :d].contiguous()
            b = input[..., d:].contiguous()
            out = out.contiguous()
            torch.ops.torch_xcpu.silu_and_mul_bf16(out, a, b)
        else:
            d = input.shape[-1] // 2
            a = input[..., :d]
            b = input[..., d:]
            out.copy_(torch.nn.functional.silu(a) * b)

        return out

    @staticmethod
    def forward_native(x: Tensor) -> Tensor:
        d = x.shape[-1] // 2
        a = x[..., :d]
        b = x[..., d:]
        return torch.nn.functional.silu(a) * b


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


def register_ops():
    try:
        load_general_plugins()
        _register_silu_and_mul_op()
        SiluAndMul.op_registry_oot["cpu"] = XcpuSiluAndMul
        logger.info("XCPU插件注册完成")
    except Exception as e:
        logger.error(f"XCPU插件注册失败：{e}")
        pass


__all__ = ["XcpuSiluAndMul", "XcpuRMSNorm", "register_ops"]
