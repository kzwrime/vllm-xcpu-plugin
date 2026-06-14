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

# Real hidden sizes from framework actual usage
HIDDEN_SIZES = [
    64,
    128,
    160,
    192,
    384,
    768,
    1024,  # Qwen3-0.6B
    2048,  # Qwen3-30B-A3B / DeepSeek-V2-Lite
    3584,  # DeepSeek-R1-Distill-Qwen-7B
    6144,  # Qwen3-Coder-480B-A35B
    7168,  # DeepSeek-V3
]

SEEDS = [0]
CUDA_DEVICES = CUSTOM_OP_TEST_DEVICES


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def _test_fused_add_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test fused_add_rms_norm with real-world framework usage."""
    set_random_seed(seed)
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cpu") * scale
    residual_cpu = torch.randn_like(x_cpu) * scale

    # Compute reference in fp32 for higher precision
    layer_fp32 = RMSNorm(hidden_size).to(dtype=torch.float)
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    x_fp32 = x_cpu.to(torch.float)
    residual_fp32 = residual_cpu.to(torch.float)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out, ref_residual = layer_fp32.forward_native(x_fp32, residual_fp32)
    x = x_cpu.to(device)
    residual = residual_cpu.to(device)
    out, new_residual = layer(x, residual)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()
    new_residual_cpu = new_residual.cpu()

    # Print error metrics
    # max_abs_error_out = (out.to(torch.float) - ref_out).abs().max().item()
    # max_rel_error_out = ((out.to(torch.float) - ref_out).abs() / (ref_out.abs() + 1e-12)).max().item()  # noqa: E501
    # max_abs_error_residual = (new_residual.to(torch.float) - ref_residual).abs().max().item()  # noqa: E501
    # max_rel_error_residual = ((new_residual.to(torch.float) - ref_residual).abs() / (ref_residual.abs() + 1e-12)).max().item()  # noqa: E501
    # print(f"  Output: max_abs_error={max_abs_error_out:.6e}, max_rel_error={max_rel_error_out:.6e}, diff_out={diff_out:.6e}")  # noqa: E501
    # print(f"  Residual: max_abs_error={max_abs_error_residual:.6e}, max_rel_error={max_rel_error_residual:.6e}, diff_residual={diff_residual:.6e}")  # noqa: E501

    # Compare using both assert_close and default_dice_tol
    # Reference precision is fp32, tolerance based on target (out) dtype
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        new_residual_cpu.to(torch.float), ref_residual, atol=atol, rtol=rtol
    )

    # Check Dice tolerance
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    diff_residual = calc_diff(new_residual_cpu.to(torch.float), ref_residual)

    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )
    assert diff_residual < default_dice_tol, (
        f"Residual diff {diff_residual} exceeds dice tolerance {default_dice_tol}"
    )

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.fused_add_rms_norm_bf16,
            (x, residual, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.fused_add_rms_norm_fp32,
            (x, residual, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")


HIDDEN_SIZES = set()
for model_name, config in ALL_MODEL_CONFIGS.items():
    if not _model_filter_matches(model_name):
        continue
    if config.hidden_size is not None:
        HIDDEN_SIZES.add(config.hidden_size)


@pytest.mark.parametrize("num_tokens", COMMON_TOKENS)
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
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cpu") * scale
    residual_cpu = torch.randn_like(x_cpu) * scale

    # Compute reference in fp32 for higher precision
    layer_fp32 = RMSNorm(hidden_size).to(dtype=torch.float)
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    x_fp32 = x_cpu.to(torch.float)
    residual_fp32 = residual_cpu.to(torch.float)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out, ref_residual = layer_fp32.forward_native(x_fp32, residual_fp32)
    x = x_cpu.to(device)
    residual = residual_cpu.to(device)
    out, new_residual = layer(x, residual)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()
    new_residual_cpu = new_residual.cpu()

    # Print error metrics
    # max_abs_error_out = (out.to(torch.float) - ref_out).abs().max().item()
    # max_rel_error_out = ((out.to(torch.float) - ref_out).abs() / (ref_out.abs() + 1e-12)).max().item()  # noqa: E501
    # max_abs_error_residual = (new_residual.to(torch.float) - ref_residual).abs().max().item()  # noqa: E501
    # max_rel_error_residual = ((new_residual.to(torch.float) - ref_residual).abs() / (ref_residual.abs() + 1e-12)).max().item()  # noqa: E501
    # print(f"  Output: max_abs_error={max_abs_error_out:.6e}, max_rel_error={max_rel_error_out:.6e}, diff_out={diff_out:.6e}")  # noqa: E501
    # print(f"  Residual: max_abs_error={max_abs_error_residual:.6e}, max_rel_error={max_rel_error_residual:.6e}, diff_residual={diff_residual:.6e}")  # noqa: E501

    # Compare using both assert_close and default_dice_tol
    # Reference precision is fp32, tolerance based on target (out) dtype
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        new_residual_cpu.to(torch.float), ref_residual, atol=atol, rtol=rtol
    )

    # Check Dice tolerance
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    diff_residual = calc_diff(new_residual_cpu.to(torch.float), ref_residual)

    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )
    assert diff_residual < default_dice_tol, (
        f"Residual diff {diff_residual} exceeds dice tolerance {default_dice_tol}"
    )

    # Check the custom kernel
    if x.dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.fused_add_rms_norm_bf16,
            (x, residual, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif x.dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.fused_add_rms_norm_fp32,
            (x, residual, layer.weight.data, layer.variance_epsilon),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        raise RuntimeError(f"Unsupported dtype: {x.dtype}")
