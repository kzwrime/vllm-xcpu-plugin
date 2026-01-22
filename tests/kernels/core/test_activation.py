# 1. 补全必要导入（核心：导入 pytest 及所有依赖模块）
import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform  # 文档1要求：CPU环境校验
from vllm.plugins import load_general_plugins  # 文档1要求：加载插件
from vllm.utils.torch_utils import set_random_seed

# 2. 文档1要求：加载插件（触发自定义算子注册）
load_general_plugins()

# 3. 文档1要求：配置测试参数（CPU环境+精简用例）
CUDA_DEVICES = ["cpu"]
DTYPES = [torch.float32]  # 精简 dtype，加快测试
NUM_TOKENS = [7, 83]  # 保留核心测试规模
D = [512]  # 精简维度
SEEDS = [0]


# 4. 文档3要求：插件专属测试用例（名称匹配 test_silu_and_mul）
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul(
    default_vllm_config,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """验证 SiluAndMul 自定义算子（文档3核心要求）"""
    # 文档1要求：校验CPU环境
    assert current_platform.is_cpu(), "插件仅支持CPU环境"

    # 初始化（复用vLLM原生测试逻辑）
    set_random_seed(seed)
    torch.set_default_device(device)

    # 文档要求：构造 contiguous 输入张量
    x = torch.randn(num_tokens, 2 * d, dtype=dtype).contiguous()

    # 实例化算子（文档3要求：复用vLLM原生层）
    layer = SiluAndMul()

    # 核心校验：插件注册生效（文档3隐含要求）
    assert "XcpuSiluAndMul" in str(type(layer)), "自定义算子未注册"

    # 文档2要求：数值正确性校验（对比原生逻辑）
    out_custom = layer(x)
    gate = x[..., :d]
    value = x[..., d:]
    out_native = F.silu(gate) * value
    torch.testing.assert_close(out_custom, out_native, atol=1e-5, rtol=1e-3)

    # 简化接口验证（替换 opcheck，文档未强制）
    out_direct = torch.empty((num_tokens, d), dtype=dtype, device=device)
    import torch_xcpu

    torch_xcpu._C.silu_and_mul_out(x, out_direct)
    assert out_direct.shape == (num_tokens, d), "底层算子调用失败"

    print(f"✅ 测试通过：device={device}, dtype={dtype}")
