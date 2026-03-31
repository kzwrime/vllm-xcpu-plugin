# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test cases for rms_norm operation.

This test file covers real-world usage from vLLM framework:
- 2D format: [num_tokens, hidden_size] for hidden state normalization
  - Regular hidden_size (2048, 3584, 6144, 7168)
  - MLA q_lora_rank (1536) and kv_lora_rank (512) for DeepSeek-V3
- 3D format: [num_tokens, num_heads, head_size] for attention head
    normalization (Q and K tensors)
  - Various num_heads from 1 to 96
  - head_size = 128
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

# For 3D format, framework only uses 1 and 28
NUM_TOKENS_3D = [
    1, 2, 4, 7, 8, 16, 31, 32, 64, 128, 133, 192, 256, 512, 577, 1024,
    2055
]

# Real hidden sizes from framework actual usage (2D format)
HIDDEN_SIZES = [
    512,    # DeepSeek-V3 kv_lora_rank
    1024,   # Qwen3-0.6B
    1536,   # DeepSeek-V3 q_lora_rank
    2048,   # Qwen3-30B-A3B
    3584,   # DeepSeek-R1-Distill-Qwen-7B
    6144,   # Qwen3-Coder-480B-A35B
    7168,   # DeepSeek-V3
]

# 3D format: [num_tokens, num_heads, head_size] for attention head normalization
# Format: (num_heads, head_size, total_size_for_stride)
# total_size_for_stride simulates the QK projection size for non-contiguous views
# Multiple total_size values for the same num_heads cover different
# stride patterns from framework
_3D_CONFIGS = [
    # num_heads=1
    (1, 128, 1),
    (1, 128, 8),
    # num_heads=2
    (2, 128, 2),
    (2, 128, 20),
    # num_heads=4
    (4, 128, 4),
    (4, 128, 16),
    (4, 128, 40),
    (4, 128, 56),
    # num_heads=6
    (6, 128, 6),
    (6, 128, 8),
    # num_heads=8
    (8, 128, 8),
    (8, 128, 16),
    (8, 128, 32),
    (8, 128, 112),
    # num_heads=16
    (16, 128, 16),
    (16, 128, 20),
    (16, 128, 32),
    # num_heads=32
    (32, 128, 32),
    (32, 128, 40),
    # num_heads=48
    (48, 128, 48),
    (48, 128, 56),
    # num_heads=96
    (96, 128, 96),
    (96, 128, 112),
]

SEEDS = [0]
CUDA_DEVICES = ["cpu"]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_2d(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test rms_norm with 2D format [num_tokens, hidden_size].

    This tests both regular hidden_size and MLA special cases:
    - hidden_size = 512: DeepSeek-V3 kv_lora (non-contiguous view)
    - hidden_size = 1536: DeepSeek-V3 q_lora (contiguous)
    - Other hidden_sizes: Regular contiguous tensors
    """
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)

    # Handle MLA kv_lora special case (non-contiguous view)
    if hidden_size == 512:
        # MLA kv_lora: non-contiguous view with stride = 512 + 64 = 576
        mla_head_size = hidden_size + 64
        combined = torch.randn(num_tokens, mla_head_size, dtype=dtype) * scale
        x = combined[:, :hidden_size]  # Non-contiguous view
    else:
        # Regular case: contiguous tensor
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale

    # Reference implementation
    ref_out = layer.forward_native(x)
    out = layer(x)

    # LayerNorm operators typically have larger numerical errors
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_bf16,
            (out, x, layer.weight.data, layer.variance_epsilon),
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_fp32,
            (out, x, layer.weight.data, layer.variance_epsilon),
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_3D)
@pytest.mark.parametrize("num_heads,head_size,total_size", _3D_CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_3d(
    default_vllm_config,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    total_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test rms_norm with 3D format [num_tokens, num_heads, head_size].

    This simulates attention head normalization for Q and K tensors.
    The framework applies rms_norm to 3D tensors directly, normalizing along
    the last dimension with weight size = head_size (not num_heads * head_size).

    The framework creates non-contiguous views by chunking a larger combined tensor.
    total_size represents the total number of heads in the combined
    tensor before chunking.
    """
    set_random_seed(seed)
    torch.set_default_device(device)

    # For 3D rms_norm, framework uses weight size = head_size
    # (not num_heads * head_size)
    # The framework applies rms_norm to 3D tensors directly, not reshaped to 2D
    layer = RMSNorm(head_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Create non-contiguous 3D view by chunking a larger combined tensor.
    # This simulates how the framework creates Q and K tensors from
    # combined QKV projections.
    if total_size > num_heads:
        # Create a larger combined tensor
        combined = torch.randn(num_tokens, total_size, head_size, dtype=dtype)
        scale = 1 / (2 * head_size)
        combined = combined * scale

        # Calculate chunk index to extract the desired num_heads slice
        # Framework uses .chunk() which creates non-contiguous views
        chunk_idx = 0  # Use the first chunk

        # Extract non-contiguous view using slicing
        # [num_tokens, chunk_idx*num_heads:(chunk_idx+1)*num_heads, head_size]
        x = combined[:, chunk_idx * num_heads:(chunk_idx + 1) * num_heads, :]
    else:
        # Contiguous case (total_size == num_heads)
        x = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
        scale = 1 / (2 * head_size)
        x = x * scale

    # Call RMSNorm layer directly on 3D tensor
    # Framework applies rms_norm to 3D tensors, normalizing along the last dimension
    out = layer(x)

    # Reference implementation
    ref_out = layer.forward_native(x)

    # LayerNorm operators typically have larger numerical errors
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_bf16,
            (out, x, layer.weight.data, layer.variance_epsilon),
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_fp32,
            (out, x, layer.weight.data, layer.variance_epsilon),
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")
