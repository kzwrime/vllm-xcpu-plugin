# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_xcpu
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.utils.torch_utils import set_random_seed

# 复用vLLM原生测试依赖
from tests.kernels.allclose_default import get_default_atol, get_default_rtol

# ========== 扩展测试用例范围：恢复原生全量参数 + 新增极端场景 ==========
# 1. 维度扩展：恢复原生全量NUM_TOKENS/D，覆盖小/中/大维度
NUM_TOKENS = [7, 83, 2048]  # 小批量/中批量/大批量
D = [64, 512, 13824]  # 小维度/常规维度/大维度（原生值）
# 2. 随机种子扩展：覆盖多个种子，避免随机值巧合
SEEDS = [0, 42, 100]
# 3. dtype保留：XCPU/CPU兼容的bf16/fp32
DTYPES = [torch.bfloat16, torch.float32]
# 4. 设备：仅保留CPU/XCPU（适配环境）
DEVICES = ["cpu"]
# 5. 新增输入类型：覆盖常规随机值、全零、极值（验证鲁棒性）
INPUT_TYPES = ["random", "all_zero", "extreme_values"]


# ========== 仅注册XCPU算子（无修改） ==========
@pytest.fixture(scope="function", autouse=True)
def register_xcpu_silu_and_mul():
    from vllm_xcpu_plugin.custom_ops import register_ops

    register_ops()
    yield


# ========== 扩展核心测试：覆盖更多维度+输入场景 ==========
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_type", INPUT_TYPES)  # 新增：输入类型参数
@torch.inference_mode()
def test_act_and_mul_silu_and_mul_xcpu_extended(
    default_vllm_config,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_type: str,
) -> None:
    """扩展版测试：覆盖更多维度+输入场景，验证XCPU算子鲁棒性"""
    # 初始化种子+设备（原生逻辑）
    set_random_seed(seed)
    torch.set_default_device(device)

    # ========== 扩展：构造不同类型的输入 ==========
    if input_type == "random":
        # 常规随机值（原生测试场景）
        x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)
    elif input_type == "all_zero":
        # 全零输入：验证算子对零值的处理
        x = torch.zeros(num_tokens, 2 * d, dtype=dtype, device=device)
    elif input_type == "extreme_values":
        # 极值输入：验证算子对大/小数值的鲁棒性
        x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)
        # 放大数值到极值范围（±1e4），同时避免inf/nan
        x = x.clamp(-1e4, 1e4) * 1e2

    # 核心：初始化层+计算输出
    layer = SiluAndMul()
    out = layer(x)
    ref_out = layer.forward_native(x)

    # 精度验证（适配XCPU float32的微小误差）
    atol = get_default_atol(out)
    rtol = get_default_rtol(out)
    if dtype == torch.float32:
        rtol = max(rtol, 1e-6)
        atol = max(atol, 1e-7)
    # 对极值输入放宽容差（数值放大后误差成比例增加，属于正常现象）
    if input_type == "extreme_values":
        rtol = max(rtol, 1e-5)
        atol = max(atol, 1e-4)

    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


# ========== 扩展降级测试：覆盖更多场景 ==========
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_type", INPUT_TYPES)  # 新增：输入类型
@torch.inference_mode()
def test_silu_and_mul_xcpu_fallback_extended(
    default_vllm_config,
    dtype: torch.dtype,
    device: str,
    input_type: str,
) -> None:
    """扩展版降级测试：覆盖不同输入类型"""
    torch.set_default_device(device)

    # 构造不同类型输入
    if input_type == "random":
        x = torch.randn(16, 1024, dtype=dtype, device=device)
    elif input_type == "all_zero":
        x = torch.zeros(16, 1024, dtype=dtype, device=device)
    elif input_type == "extreme_values":
        x = torch.randn(16, 1024, dtype=dtype, device=device)
        x = x.clamp(-1e4, 1e4) * 1e2

    # 模拟XCPU算子不可用
    original_fp32 = getattr(torch_xcpu._C, "silu_and_mul_fp32", None)
    original_bf16 = getattr(torch_xcpu._C, "silu_and_mul_bf16", None)
    torch_xcpu._C.silu_and_mul_fp32 = None
    torch_xcpu._C.silu_and_mul_bf16 = None

    # 验证降级逻辑
    layer = SiluAndMul()
    out = layer(x)
    ref_out = layer.forward_native(x)

    # 精度验证
    atol = get_default_atol(out)
    rtol = get_default_rtol(out)
    if input_type == "extreme_values":
        rtol = max(rtol, 1e-5)
        atol = max(atol, 1e-4)

    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)

    # 恢复算子
    if original_fp32:
        torch_xcpu._C.silu_and_mul_fp32 = original_fp32
    if original_bf16:
        torch_xcpu._C.silu_and_mul_bf16 = original_bf16
