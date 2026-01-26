import torch
from torch import Tensor  # 显式导入Tensor类型，解决mypy类型注解问题
from vllm.logger import logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform


# 工具函数：检查torch.ops.torch_xcpu算子是否可用（替换原._C检查）
def _is_torch_xcpu_op_available(op_name: str) -> bool:
    try:
        # 改为检查torch.ops.torch_xcpu下的算子
        return hasattr(torch.ops.torch_xcpu, op_name)
    except Exception:
        return False


# 清空原生注册，避免重复
if hasattr(SiluAndMul, "op_registry_oot"):
    SiluAndMul.op_registry_oot = {}
    logger.info("清空原生SiluAndMul的op_registry_oot，避免重复注册")


# 注册torch.ops算子（移除强制assert，适配torch.ops路径）
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

                # 替换：从torch_xcpu._C改为torch.ops.torch_xcpu
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


# 核心：移除forward_cpu中的所有logger语句 + 修复forward_native签名
@SiluAndMul.register_oot
class XcpuSiluAndMul(SiluAndMul):
    def __init__(self) -> None:
        super().__init__()
        # __init__里的日志不会被编译，可保留
        logger.info("Init XcpuSiluAndMul (XCPU backend)")

        if current_platform.is_cpu():
            try:
                self._forward_method = self.forward_cpu
                fp32_ok = _is_torch_xcpu_op_available("silu_and_mul_fp32")
                bf16_ok = _is_torch_xcpu_op_available("silu_and_mul_bf16")
                # 仅保留__init__里的日志（非编译路径）
                logger.info(f"XCPU算子可用性：fp32={fp32_ok}, bf16={bf16_ok}")
            except Exception as e:
                logger.warning(f"Failed to set forward_cpu, fallback to native: {e}")
                self._forward_method = self.forward_native
        else:
            self._forward_method = self.forward_native

    def forward_cpu(self, input: Tensor) -> Tensor:
        """移除所有logger语句，避免torch.compile不兼容"""
        # 1. 彻底删除logger.debug语句（核心修复）
        # 2. 保留核心逻辑，无任何日志输出
        if input.size(-1) % 2 != 0:
            raise ValueError(
                f"SiluAndMul input last dim must be even, got {input.size(-1)}"
            )

        d_out = input.shape[-1] // 2
        out = torch.empty(
            *input.shape[:-1], d_out, dtype=input.dtype, device=input.device
        )

        # 替换：从torch_xcpu._C改为torch.ops.torch_xcpu
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

    # 核心修复：对齐父类签名（添加@staticmethod + 参数名改为x）
    @staticmethod
    def forward_native(x: Tensor) -> Tensor:
        """
        修复mypy override错误：
        1. 添加@staticmethod装饰器（对齐父类）
        2. 参数名改为x（对齐父类的forward_native(x: Tensor)）
        3. 内部逻辑保持不变，仅把input替换为x
        """
        d = x.shape[-1] // 2
        a = x[..., :d]
        b = x[..., d:]
        return torch.nn.functional.silu(a) * b


def register_ops():
    try:
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        _register_silu_and_mul_op()
        SiluAndMul.op_registry_oot["cpu"] = XcpuSiluAndMul
        logger.info("XCPU插件注册完成")
    except Exception as e:
        logger.error(f"XCPU插件注册失败：{e}")
        pass


__all__ = ["XcpuSiluAndMul", "register_ops"]