# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import pytest
import torch
import torch_xcpu  # noqa: F401
from ops_test_data import case_id, run_data_mode_case
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


def _activation_layer_and_fn(
    activation: str,
    x: torch.Tensor,
    threshold: float | None,
):
    if activation == "silu_and_mul":
        layer = SiluAndMul()
        fn = (
            torch.ops.torch_xcpu.silu_and_mul_bf16
            if x.dtype == torch.bfloat16
            else torch.ops.torch_xcpu.silu_and_mul_fp32
        )
    elif activation == "mul_and_silu":
        layer = MulAndSilu()
        fn = torch.ops._C.mul_and_silu
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
        fn = torch.ops._C.gelu_and_mul
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
        fn = torch.ops._C.gelu_tanh_and_mul
    elif activation == "fatrelu":
        assert threshold is not None
        layer = FatreluAndMul(threshold)
        fn = torch.ops._C.fatrelu_and_mul
    elif activation == "swigluoai_and_mul":
        layer = SwigluOAIAndMul()
        fn = torch.ops._C.swigluoai_and_mul
    else:
        raise RuntimeError(f"Unsupported activation: {activation}")
    return layer, fn


def _run_activation_case(
    case: dict,
    activation: str,
    device: str,
) -> torch.Tensor:
    inputs = case["inputs"]
    x = inputs["x"].to(device)
    layer, fn = _activation_layer_and_fn(activation, x, inputs.get("threshold"))
    out = layer(x)
    torch.accelerator.synchronize()
    out_cpu = out.cpu()

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    opcheck_out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(
            fn,
            (opcheck_out, x, inputs["threshold"]),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif activation == "swigluoai_and_mul":
        opcheck(
            fn,
            (opcheck_out, x, layer.alpha, layer.limit),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    else:
        opcheck(fn, (opcheck_out, x), cond=CUSTOM_OP_TEST_ENABLE_OPCHECK)
    return out_cpu


def _check_activation_case(out_cpu: torch.Tensor, case: dict) -> None:
    ref_out = case["expected"]
    atol = get_default_atol(out_cpu)
    rtol = get_default_rtol(out_cpu)
    torch.testing.assert_close(out_cpu.to(torch.float), ref_out, atol=atol, rtol=rtol)
    diff_out = calc_diff(out_cpu.to(torch.float), ref_out)
    assert diff_out < default_dice_tol, (
        f"Output diff {diff_out} exceeds dice tolerance {default_dice_tol}"
    )


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
    request: pytest.FixtureRequest,
) -> None:
    def build_case():
        set_random_seed(seed)
        x_cpu = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cpu")
        threshold = random.uniform(0, 1) if activation == "fatrelu" else None
        layer, _ = _activation_layer_and_fn(activation, x_cpu, threshold)
        ref_out = layer.to(dtype=torch.float).forward_native(x_cpu.to(torch.float))
        inputs = {"x": x_cpu}
        if threshold is not None:
            inputs["threshold"] = threshold
        return {"inputs": inputs, "expected": ref_out}

    run_data_mode_case(
        op_name="activation",
        case_name=case_id(activation, f"tokens={num_tokens}", f"d={d}", dtype, seed),
        build_fn=build_case,
        run_fn=lambda case: _run_activation_case(case, activation, device),
        check_fn=_check_activation_case,
        pytest_config=request.config,
    )


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
