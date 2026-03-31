# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test cases for fused_add_rms_norm operation.

This test file covers real-world usage from vLLM framework:
- Only hidden_size is tested (framework doesn't use fused_add_rms_norm on other sizes)
- Only bfloat16 dtype is tested (framework only uses bfloat16)
- Token counts: 1 (decode) and various prefill sizes
"""

import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.plugins import load_general_plugins
from vllm.utils.torch_utils import set_random_seed

from tests.kernels.utils import opcheck

load_general_plugins()

# Framework only uses bfloat16
DTYPES = [torch.bfloat16]

# Real token counts from framework: 1 (decode) and various prefill sizes
NUM_TOKENS = [1, 2, 4, 7, 8, 16, 31, 32, 64, 128, 133, 192, 256, 512, 577, 1024, 2055]

# Real hidden sizes from framework actual usage
HIDDEN_SIZES = [
    1024,   # Qwen3-0.6B
    2048,   # Qwen3-30B-A3B / DeepSeek-V2-Lite
    3584,   # DeepSeek-R1-Distill-Qwen-7B
    6144,   # Qwen3-Coder-480B-A35B
    7168,   # DeepSeek-V3
]

SEEDS = [0]
CUDA_DEVICES = ["cpu"]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fused_add_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test fused_add_rms_norm with real-world framework usage."""
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    residual = torch.randn_like(x) * scale

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out, ref_residual = layer.forward_native(x, residual)
    out, new_residual = layer(x, residual)

    # LayerNorm operators typically have larger numerical errors
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(new_residual, ref_residual, atol=1e-2, rtol=1e-2)

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.fused_add_rms_norm_bf16,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.fused_add_rms_norm_fp32,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")
