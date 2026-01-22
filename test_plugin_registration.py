import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins

# 1. 必须加载插件（触发自定义算子注册）
load_general_plugins()

# 2. 配置为CPU环境（文档要求，仅运行CPU用例）
CUDA_DEVICES = ["cpu"]
# 3. 简化测试参数（快速验证）
HIDDEN_SIZES = [8, 16]
DTYPES = [torch.float32, torch.float16]


# 4. 用例命名需包含 "test_silu_and_mul"（匹配过滤关键词）
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_silu_and_mul_xcpu(default_vllm_config, dtype, hidden_size):
    """测试自定义SiluAndMul算子（CPU）"""
    # 确保当前平台是CPU
    assert current_platform.is_cpu(), "测试仅支持CPU环境"

    # 创建输入张量（shape [..., 2*d]，符合算子要求）
    input = torch.randn(2, 4, 2 * hidden_size, dtype=dtype, device="cpu").contiguous()

    # 实例化算子（依赖default_vllm_config fixture设置上下文）
    silu_and_mul = SiluAndMul()
    # 验证是自定义实现
    assert isinstance(silu_and_mul, type("XcpuSiluAndMul", (), {})), "未加载自定义算子"

    # 调用自定义算子
    out = silu_and_mul(input)

    # 验证输出形状（最后一维减半）
    expected_shape = list(input.shape)
    expected_shape[-1] = expected_shape[-1] // 2
    assert out.shape == tuple(expected_shape), "输出形状错误"

    # 与原生逻辑对比（验证数值正确性）
    gate = input[..., :hidden_size]
    value = input[..., hidden_size:]
    expected = torch.nn.functional.silu(gate) * value
    torch.testing.assert_close(out, expected, atol=1e-5)
    print("✅ 自定义SiluAndMul算子测试通过（CPU）")
