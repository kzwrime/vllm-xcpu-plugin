# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import pytest
import torch
import torch_xcpu  # noqa: F401
from torch_xcpu.model_configs import ALL_MODEL_CONFIGS, COMMON_TOKENS
from vllm.model_executor.layers.activation import (
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    SiluAndMul,
    SwigluOAIAndMul,
)
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

DTYPES = [torch.bfloat16, torch.float]
NUM_TOKENS = [1, 2, 4, 7, 8, 16, 31, 32, 64, 128, 133, 192, 256, 512, 577, 1024, 2055]
NUM_TOKENS = COMMON_TOKENS
D = set([512, 13824])  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = CUSTOM_OP_TEST_DEVICES
# CUDA_DEVICES = [
#     f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
# ]


def _model_filter_matches(model_name: str) -> bool:
    model_filter = os.getenv("TEST_MODELS")
    if not model_filter:
        return True
    return any(pattern.strip() in model_name for pattern in model_filter.split(","))


for model_name, config in ALL_MODEL_CONFIGS.items():
    if not _model_filter_matches(model_name):
        continue
    if config.is_moe:
        # MoE models: use moe_intermediate_size
        width = config.moe_intermediate_size
        assert width is not None, (
            f"MoE model {model_name} must have moe_intermediate_size defined"
        )
        D.add(width)
        if config.n_shared_experts is not None:
            D.add(width * config.n_shared_experts)

    if config.intermediate_size is not None:
        # Dense models: consider TP configurations (width is divided by tp_size)
        base_width = config.intermediate_size
        if not config.tp_sizes:
            # No TP config, use base width
            D.add(base_width)
        else:
            for tp_size in config.tp_sizes:
                width = base_width // tp_size
                label = f"{model_name}_tp{tp_size}"
                D.add(width)


@pytest.mark.parametrize(
    "activation",
    [
        "silu_and_mul",
        # "mul_and_silu",
        # "gelu",
        # "gelu_tanh",
        # "fatrelu",
        # "swigluoai_and_mul",
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    default_vllm_config,
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    x_cpu = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cpu")
    x = x_cpu.to(device)
    if activation == "silu_and_mul":
        layer = SiluAndMul()
        fn = torch.ops._C.silu_and_mul
    if activation == "mul_and_silu":
        layer = MulAndSilu()
        fn = torch.ops._C.mul_and_silu
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
        fn = torch.ops._C.gelu_and_mul
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
        fn = torch.ops._C.gelu_tanh_and_mul
    elif activation == "fatrelu":
        threshold = random.uniform(0, 1)
        layer = FatreluAndMul(threshold)
        fn = torch.ops._C.fatrelu_and_mul
    elif activation == "swigluoai_and_mul":
        layer = SwigluOAIAndMul()
        fn = torch.ops._C.swigluoai_and_mul

    # Compute reference in fp32 for higher precision
    x_fp32 = x_cpu.to(torch.float)
    layer_fp32 = layer.to(dtype=torch.float)
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

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "silu_and_mul":
        fn = (
            torch.ops.torch_xcpu.silu_and_mul_bf16
            if x.dtype == torch.bfloat16
            else torch.ops.torch_xcpu.silu_and_mul_fp32
        )
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold), cond=CUSTOM_OP_TEST_ENABLE_OPCHECK)
    elif activation == "swigluoai_and_mul":
        opcheck(
            fn, (out, x, layer.alpha, layer.limit), cond=CUSTOM_OP_TEST_ENABLE_OPCHECK
        )
    else:
        opcheck(fn, (out, x), cond=CUSTOM_OP_TEST_ENABLE_OPCHECK)


# @pytest.mark.parametrize(
#     "activation",
#     [
#         (FastGELU, torch.ops._C.gelu_fast),
#         (NewGELU, torch.ops._C.gelu_new),
#         (QuickGELU, torch.ops._C.gelu_quick),
#     ],
# )
# @pytest.mark.parametrize("num_tokens", NUM_TOKENS)
# @pytest.mark.parametrize("d", D)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# @torch.inference_mode()
# def test_activation(
#     default_vllm_config,
#     activation: type[torch.nn.Module],
#     num_tokens: int,
#     d: int,
#     dtype: torch.dtype,
#     seed: int,
#     device: str,
# ) -> None:
#     set_random_seed(seed)
#     torch.set_default_device(device)
#     x = torch.randn(num_tokens, d, dtype=dtype)
#     layer = activation[0]()
#     fn = activation[1]
#     out = layer(x)
#     ref_out = layer.forward_native(x)
#     torch.testing.assert_close(
#         out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
#     )

#     out = torch.empty_like(x)
#     opcheck(fn, (out, x))
