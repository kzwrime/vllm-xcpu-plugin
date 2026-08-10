# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_xcpu  # noqa: F401
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_layer_norm(dtype: torch.dtype, device: str, default_vllm_config) -> None:
    x_cpu = torch.randn(7, 128, dtype=dtype)
    layer = LayerNorm(128).to(device=device)
    assert layer.__class__.__name__ == "XcpuLayerNorm"
    layer.weight.data.normal_(mean=1.0, std=0.1)
    layer.bias.data.normal_(mean=0.0, std=0.1)

    expected = torch.nn.functional.layer_norm(
        x_cpu.float(),
        (128,),
        layer.weight.cpu(),
        layer.bias.cpu(),
        layer.eps,
    ).to(dtype)
    actual = layer(x_cpu.to(device)).cpu()

    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_layer_norm_row_strided(
    dtype: torch.dtype, device: str, default_vllm_config
) -> None:
    storage_cpu = torch.randn(7, 129, dtype=dtype)
    x_cpu = storage_cpu[:, :128]
    assert x_cpu.stride() == (129, 1)
    layer = LayerNorm(128).to(device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    layer.bias.data.normal_(mean=0.0, std=0.1)
    expected = layer.forward_native(x_cpu.to(device)).cpu()
    actual = layer(storage_cpu.to(device)[:, :128]).cpu()
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


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
) -> None:
    set_random_seed(seed)
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=device)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base_cpu = torch.randn(num_tokens, last_dim, dtype=dtype, device="cpu")
    x_cpu = x_base_cpu[..., :hidden_size]
    assert x_cpu.is_contiguous() != strided_input
    x_cpu *= scale
    residual_cpu = torch.randn_like(x_cpu) * scale if add_residual else None
    x_base = x_base_cpu.to(device)
    x = x_base[..., :hidden_size]
    assert x.stride() == x_cpu.stride()
    residual = residual_cpu.to(device) if residual_cpu is not None else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    layer_fp32 = RMSNorm(hidden_size).to(dtype=torch.float, device="cpu")
    layer_fp32.weight.data = layer.weight.data.cpu().to(torch.float)
    ref_out = layer_fp32.forward_native(
        x_cpu.to(torch.float),
        residual_cpu.to(torch.float) if residual_cpu is not None else None,
    )
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(
            out[0].cpu().to(torch.float), ref_out[0], atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            out[1].cpu().to(torch.float), ref_out[1], atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(
            out.cpu().to(torch.float), ref_out, atol=1e-2, rtol=1e-2
        )

    if residual is not None:
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
