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
import torch_xcpu  # noqa: F401
from test_activation import _model_filter_matches
from torch_xcpu.model_configs import ALL_MODEL_CONFIGS, COMMON_TOKENS
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.plugins import load_general_plugins
from vllm.utils.torch_utils import set_random_seed

from tests.kernels.allclose_default import (
    calc_diff,
    default_dice_tol,
    get_default_atol,
    get_default_rtol,
)
from tests.kernels.utils import (
    CUSTOM_OP_TEST_DEVICES,
    CUSTOM_OP_TEST_ENABLE_OPCHECK,
    opcheck,
)

load_general_plugins()

# Framework only uses bfloat16
DTYPES = [torch.bfloat16]

# Real token counts from framework: 1 (decode) and various prefill sizes
NUM_TOKENS = [1, 2, 4, 7, 8, 16, 31, 32, 64, 128, 133, 192, 256, 512, 577, 1024, 2055]

# For 3D format, framework only uses 1 and 28
NUM_TOKENS_3D = [
    1,
    2,
    4,
    7,
    8,
    16,
    31,
    32,
    64,
    128,
    133,
    192,
    256,
    512,
    577,
    1024,
    2055,
]

# Real hidden sizes from framework actual usage (2D format)
HIDDEN_SIZES = [
    512,  # DeepSeek-V3 kv_lora_rank
    1024,  # Qwen3-0.6B
    1536,  # DeepSeek-V3 q_lora_rank
    2048,  # Qwen3-30B-A3B
    3584,  # DeepSeek-R1-Distill-Qwen-7B
    6144,  # Qwen3-Coder-480B-A35B
    7168,  # DeepSeek-V3
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
CUDA_DEVICES = CUSTOM_OP_TEST_DEVICES


# ---------------------------------------------------------------------------
# Model config-based test cases (following test_activation.py pattern)
# ---------------------------------------------------------------------------

# Collect hidden sizes and 3D configs from model configurations
MODEL_HIDDEN_SIZES = set()
MODEL_3D_CONFIGS = []

for model_name, config in ALL_MODEL_CONFIGS.items():
    if not _model_filter_matches(model_name):
        continue
    # 2D format: hidden_size
    if config.hidden_size is not None:
        MODEL_HIDDEN_SIZES.add((config.hidden_size, False, False))

    # DeepSeek-V3 special cases: q_lora_rank and kv_lora_rank
    if config.kv_lora_rank is not None:
        if config.q_lora_rank is not None:
            MODEL_HIDDEN_SIZES.add((config.q_lora_rank, True, True))
            MODEL_HIDDEN_SIZES.add((
                config.kv_lora_rank,
                True,
                True,
            ))  # is_mla_kv_lora = True, has_q_lora = True
        else:
            MODEL_HIDDEN_SIZES.add((
                config.kv_lora_rank,
                True,
                False,
            ))  # is_mla_kv_lora = True, has_q_lora = False

    # 3D format: [num_tokens, num_heads, head_size]
    for tp_size in config.tp_sizes:
        if config.num_heads is not None and config.head_size is not None:
            num_heads_per_rank = max(config.num_heads // tp_size, 1)
            MODEL_3D_CONFIGS.append((
                num_heads_per_rank,
                config.head_size,
                model_name,
                tp_size,
                "Q",
            ))

        # For GQA models: also test K tensor normalization
        if config.is_gqa:
            num_kv_heads = config.num_kv_heads
            if num_kv_heads is not None:
                num_kv_heads_per_rank = max(num_kv_heads // tp_size, 1)
                MODEL_3D_CONFIGS.append((
                    num_kv_heads_per_rank,
                    config.head_size,
                    model_name,
                    tp_size,
                    "KV",
                ))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def _test_rms_norm_2d(
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
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)

    # Handle MLA kv_lora special case (non-contiguous view)
    if hidden_size == 512:
        # MLA kv_lora: non-contiguous view with stride = 512 + 64 = 576
        mla_head_size = hidden_size + 64
        combined = (
            torch.randn(num_tokens, mla_head_size, dtype=dtype, device="cpu") * scale
        )
        x_cpu = combined[:, :hidden_size]  # Non-contiguous view
        x = combined.to(device)[:, :hidden_size]
        assert x.stride() == x_cpu.stride()
    else:
        # Regular case: contiguous tensor
        x_cpu = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cpu") * scale
        x = x_cpu.to(device)

    # Compute reference in fp32 for higher precision
    layer_fp32 = RMSNorm(hidden_size).to(dtype=torch.float)
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    x_fp32 = x_cpu.to(torch.float)
    ref_out = layer_fp32.forward_native(x_fp32)

    out = layer(x)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()

    # Print error metrics
    # max_abs_error = (out.to(torch.float) - ref_out).abs().max().item()
    # max_rel_error = ((out.to(torch.float) - ref_out).abs() / (ref_out.abs() + 1e-12)).max().item()  # noqa: E501
    # print(f"  Output: max_abs_error={max_abs_error:.6e}, max_rel_error={max_rel_error:.6e}, diff_out={diff_out:.6e}")  # noqa: E501

    # Compare using both assert_close and default_dice_tol
    # Reference precision is fp32, tolerance based on target (out) dtype
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)

    # Check Dice tolerance
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_bf16,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_fp32,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_3D)
@pytest.mark.parametrize("num_heads,head_size,total_size", _3D_CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def _test_rms_norm_3d(
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

    # For 3D rms_norm, framework uses weight size = head_size
    # (not num_heads * head_size)
    # The framework applies rms_norm to 3D tensors directly, not reshaped to 2D
    layer = RMSNorm(head_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Create non-contiguous 3D view by chunking a larger combined tensor.
    # This simulates how the framework creates Q and K tensors from
    # combined QKV projections.
    if total_size > num_heads:
        # Create a larger combined tensor
        combined = torch.randn(
            num_tokens, total_size, head_size, dtype=dtype, device="cpu"
        )
        scale = 1 / (2 * head_size)
        combined = combined * scale

        # Calculate chunk index to extract the desired num_heads slice
        # Framework uses .chunk() which creates non-contiguous views
        chunk_idx = 0  # Use the first chunk

        # Extract non-contiguous view using slicing
        # [num_tokens, chunk_idx*num_heads:(chunk_idx+1)*num_heads, head_size]
        x_cpu = combined[:, chunk_idx * num_heads : (chunk_idx + 1) * num_heads, :]
        x = combined.to(device)[
            :, chunk_idx * num_heads : (chunk_idx + 1) * num_heads, :
        ]
        assert x.stride() == x_cpu.stride()
    else:
        # Contiguous case (total_size == num_heads)
        x_cpu = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cpu")
        scale = 1 / (2 * head_size)
        x_cpu = x_cpu * scale
        x = x_cpu.to(device)

    # Compute reference in fp32 for higher precision
    layer_fp32 = RMSNorm(head_size).to(dtype=torch.float)
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    x_fp32 = x_cpu.to(torch.float)
    ref_out = layer_fp32.forward_native(x_fp32)

    # Call RMSNorm layer directly on 3D tensor
    # Framework applies rms_norm to 3D tensors, normalizing along the last dimension
    out = layer(x)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()

    # Print error metrics
    # max_abs_error = (out.to(torch.float) - ref_out).abs().max().item()
    # max_rel_error = ((out.to(torch.float) - ref_out).abs() / (ref_out.abs() + 1e-12)).max().item()  # noqa: E501
    # print(f"  Output: max_abs_error={max_abs_error:.6e}, max_rel_error={max_rel_error:.6e}, diff_out={diff_out:.6e}")  # noqa: E501

    # Compare using both assert_close and default_dice_tol
    # Reference precision is fp32, tolerance based on target (out) dtype
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)

    # Check Dice tolerance
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_bf16,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_fp32,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")


# ---------------------------------------------------------------------------
# Model config-based tests (following test_activation.py pattern)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", COMMON_TOKENS)
@pytest.mark.parametrize(
    "hidden_size,is_mla_kv_lora,has_q_lora", sorted(MODEL_HIDDEN_SIZES)
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_2d_model_configs(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    is_mla_kv_lora: bool,
    has_q_lora: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test rms_norm with 2D format [num_tokens, hidden_size] using model configs.

    Tests hidden sizes derived from actual model configurations,
    including DeepSeek-V3 q_lora_rank and kv_lora_rank special cases.
    """
    set_random_seed(seed)
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)

    if is_mla_kv_lora:
        # MLA kv_lora: non-contiguous view with stride = kv_lora_rank + qk_rope_head_dim
        kv_lora_rank = 512
        qk_rope_head_dim = 64  # DeepSeek-V3 rotary_dim
        if has_q_lora:
            q_lora_rank = 1536
            mla_head_size = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        else:
            mla_head_size = kv_lora_rank + qk_rope_head_dim
        combined = (
            torch.randn(num_tokens, mla_head_size, dtype=dtype, device="cpu") * scale
        )
        x_cpu = combined[:, :hidden_size]  # Non-contiguous view
        x = combined.to(device)[:, :hidden_size]
        assert x.stride() == x_cpu.stride()
    else:
        # Regular case: contiguous tensor
        x_cpu = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cpu") * scale
        x = x_cpu.to(device)

    # Compute reference in fp32 for higher precision
    layer_fp32 = RMSNorm(hidden_size).to(dtype=torch.float)
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    x_fp32 = x_cpu.to(torch.float)
    ref_out = layer_fp32.forward_native(x_fp32)

    out = layer(x)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()

    # Compare using both assert_close and default_dice_tol
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)

    # Check Dice tolerance
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_bf16,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_fp32,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")


@pytest.mark.parametrize("num_tokens", COMMON_TOKENS)
@pytest.mark.parametrize(
    "num_heads,head_size,model_name,tp_size,tensor_type", MODEL_3D_CONFIGS
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_3d_model_configs(
    default_vllm_config,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    model_name: str,
    tp_size: int,
    tensor_type: str,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test rms_norm with 3D format using model configs.

    Format: [num_tokens, num_heads, head_size].
    Tests attention head normalization for Q and K tensors
    with realistic num_heads derived from model configurations and TP sizes.
    """
    set_random_seed(seed)

    config = ALL_MODEL_CONFIGS[model_name]

    # For 3D rms_norm, framework uses weight size = head_size
    layer = RMSNorm(head_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1 / (2 * head_size)

    # Create non-contiguous 3D view simulating framework behavior
    if num_tokens > 1:
        # Multi-token: simulate non-contiguous input from QK projection view
        complete_num_kv_heads = config.num_kv_heads if config.num_kv_heads else 0
        complete_num_heads = config.num_heads

        if complete_num_kv_heads > 0:
            num_kv_heads_per_rank = max(complete_num_kv_heads // tp_size, 1)
        else:
            num_kv_heads_per_rank = 0

        # Calculate total_size for stride based on tensor type
        if tensor_type == "KV":
            # K tensor: stride based on Q heads (num_heads_per_rank from Q side)
            num_heads_per_rank_q = max(config.num_heads // tp_size, 1)
            total_size = num_heads_per_rank_q + 2 * num_kv_heads_per_rank
        else:
            # Q tensor
            if config.num_heads is not None:
                num_heads_per_rank_q = max(config.num_heads // tp_size, 1)
            else:
                num_heads_per_rank_q = num_heads

            if complete_num_heads is not None and tp_size == 1:
                total_size = complete_num_heads + 2 * num_kv_heads_per_rank
            elif num_heads_per_rank_q is not None:
                total_size = num_heads_per_rank_q + 2 * num_kv_heads_per_rank
            else:
                total_size = num_heads + 2 * num_kv_heads_per_rank

        combined = (
            torch.randn(num_tokens, total_size, head_size, dtype=dtype, device="cpu")
            * scale
        )
        x_cpu = combined[:, :num_heads, :]  # Non-contiguous view
        x = combined.to(device)[:, :num_heads, :]
        assert x.stride() == x_cpu.stride()
    else:
        # Single token: contiguous (decode mode)
        x_cpu = (
            torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cpu")
            * scale
        )
        x = x_cpu.to(device)

    # Compute reference in fp32 for higher precision
    layer_fp32 = RMSNorm(head_size).to(dtype=torch.float)
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    x_fp32 = x_cpu.to(torch.float)
    ref_out = layer_fp32.forward_native(x_fp32)

    out = layer(x)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()

    # Compare using both assert_close and default_dice_tol
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)

    # Check Dice tolerance
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_bf16,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rms_norm_fp32,
            (out, x, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")
