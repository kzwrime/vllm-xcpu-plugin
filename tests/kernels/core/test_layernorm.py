# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.plugins import load_general_plugins
from vllm.utils.torch_utils import set_random_seed

from tests.kernels.utils import opcheck

load_general_plugins()

DTYPES = [torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing

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
    1792,
    2048,
    2432,
    2560,
    2736,
    2816,
    3072,
    3584,
    4096,
    4608,
    4736,
    4864,
    5120,
    5128,
    5472,
    5632,
    6144,
    6400,
    7168,
    8192,
    8200,
    9216,
    9472,
    9728,
    10944,
    11264,
    12288,
    12800,
    16384,
    18432,
    18944,
    19456,
    21888,
    24576,
    25600,
    36864,
    37888,
    51200,
]

ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = ["cpu"]
# CUDA_DEVICES = [
#     f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
# ]


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
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
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
