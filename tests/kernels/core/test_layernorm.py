# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_xcpu  # noqa: F401
from ops_test_data import case_id, run_data_mode_case
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.plugins import load_general_plugins
from vllm.utils.torch_utils import set_random_seed

from tests.kernels.utils import (
    CUSTOM_OP_TEST_DEVICES,
    CUSTOM_OP_TEST_ENABLE_OPCHECK,
    opcheck,
)

load_general_plugins()

DTYPES = [torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 333]  # Arbitrary values for testing

# fmt: skip
HIDDEN_SIZES = [
    8,
    192,
    352,
    384,
    512,
    704,
    768,
    776,
    896,
    1024,
    1280,
    1408,
    1536,
]

ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = CUSTOM_OP_TEST_DEVICES
# CUDA_DEVICES = [
#     f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
# ]


def _run_rms_norm_case(case: dict, hidden_size: int, dtype: torch.dtype, device: str):
    inputs = case["inputs"]
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=device)
    layer.weight.data = inputs["weight"].to(device)
    x = inputs["x"].to(device)
    residual_cpu = inputs.get("residual")
    residual = residual_cpu.to(device) if residual_cpu is not None else None
    out = layer(x, residual)
    torch.accelerator.synchronize()

    if residual is not None:
        actual = {"out": out[0].cpu(), "residual": out[1].cpu()}
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
        actual = {"out": out.cpu()}
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
    return actual


def _check_rms_norm_case(actual: dict, case: dict) -> None:
    expected = case["expected"]
    torch.testing.assert_close(
        actual["out"].to(torch.float), expected["out"], atol=1e-2, rtol=1e-2
    )
    if "residual" in expected:
        torch.testing.assert_close(
            actual["residual"].to(torch.float),
            expected["residual"],
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
    request: pytest.FixtureRequest,
) -> None:
    def build_case():
        set_random_seed(seed)
        layer = RMSNorm(hidden_size).to(dtype=dtype, device="cpu")
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        last_dim = 2 * hidden_size if strided_input else hidden_size
        x_base_cpu = torch.randn(num_tokens, last_dim, dtype=dtype, device="cpu")
        x_cpu = x_base_cpu[..., :hidden_size]
        assert x_cpu.is_contiguous() != strided_input
        x_cpu *= scale
        residual_cpu = torch.randn_like(x_cpu) * scale if add_residual else None

        layer_fp32 = RMSNorm(hidden_size).to(dtype=torch.float, device="cpu")
        layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
        ref_out = layer_fp32.forward_native(
            x_cpu.to(torch.float),
            residual_cpu.to(torch.float) if residual_cpu is not None else None,
        )
        expected = (
            {"out": ref_out[0], "residual": ref_out[1]}
            if add_residual
            else {"out": ref_out}
        )
        inputs = {"x": x_cpu, "weight": layer.weight.data.cpu()}
        if residual_cpu is not None:
            inputs["residual"] = residual_cpu
        return {"inputs": inputs, "expected": expected}

    run_data_mode_case(
        op_name="layernorm_rms_norm",
        case_name=case_id(
            f"tokens={num_tokens}",
            f"hidden={hidden_size}",
            f"residual={add_residual}",
            dtype,
            f"seed={seed}",
            f"strided={strided_input}",
        ),
        build_fn=build_case,
        run_fn=lambda case: _run_rms_norm_case(case, hidden_size, dtype, device),
        check_fn=_check_rms_norm_case,
        pytest_config=request.config,
    )
