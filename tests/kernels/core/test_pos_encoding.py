# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._prims_common import TensorLikeType
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.plugins import load_general_plugins
from vllm.utils.torch_utils import set_random_seed

load_general_plugins()

# =============================================================================
# Copied from tests/kernels/allclose_default.py to avoid import issues
# =============================================================================
default_atol = {torch.float16: 1e-3, torch.bfloat16: 2e-2, torch.float: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float: 1.3e-6}


def get_default_atol(output) -> float:
    return default_atol[output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]


# =============================================================================
# Copied from tests/kernels/utils.py to avoid import issues
# =============================================================================
ALL_OPCHECK_TEST_UTILS: tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """Reference implementation of torch.allclose"""
    torch._refs._check_close_args(name="torch.allclose", a=a, b=b, rtol=rtol, atol=atol)
    return bool(
        torch.all(
            torch.isclose(
                a.double(), b.double(), rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        ).item()
    )


def opcheck(
    op: (
        torch._ops.OpOverload
        | torch._ops.OpOverloadPacket
        | torch._library.custom_ops.CustomOpDef
    ),
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    test_utils: str | Sequence[str] = ALL_OPCHECK_TEST_UTILS,
    raise_exception: bool = True,
    cond: bool = True,
) -> dict[str, str]:
    with patch("torch.allclose", new=fp8_allclose):
        return (
            torch.library.opcheck(
                op, args, kwargs, test_utils=test_utils, raise_exception=raise_exception
            )
            if cond
            else {}
        )


# =============================================================================
# Test Configuration
# =============================================================================
IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float]
# Modified to cover Qwen2/Qwen3 (128 is typical for 7B/72B, 64 for 0.5B)
HEAD_SIZES = [64, 80, 120, 128, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [16, 17]
BATCH_SIZES = [1, 2, 4, 8, 16, 5, 13]
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = ["cpu"]
USE_KEY = [True, False]


def _get_flat_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads * head_size)


# For testing sliced tensors
def _get_padded_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size + 64)


def _get_batch_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size)


TENSORS_SHAPES_FN = [
    _get_batch_tensor_shape,
    _get_flat_tensor_shape,
    _get_padded_tensor_shape,
]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSORS_SHAPES_FN)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_key", USE_KEY)
@torch.inference_mode()
def test_rotary_embedding(
    default_vllm_config,
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int, ...]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: int | None,
    dtype: torch.dtype,
    seed: int,
    device: str,
    use_key: bool,
    max_position: int = 8192,
    rope_theta: float = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size

    set_random_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": rotary_dim / head_size,
    }
    rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype)
    key = torch.randn_like(query) if use_key else None

    # slice tensor if required, noop otherwise
    query = query[..., :head_size]
    key = key[..., :head_size] if use_key else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(
        positions, query.clone(), key.clone() if key is not None else None
    )
    out_query, out_key = rope.forward(positions, query, key)

    # Compare the results.
    torch.testing.assert_close(
        out_query,
        ref_query,
        atol=get_default_atol(out_query),
        rtol=get_default_rtol(out_query),
    )
    if use_key:
        torch.testing.assert_close(
            out_key,
            ref_key,
            atol=get_default_atol(out_key),
            rtol=get_default_rtol(out_key),
        )
    else:
        assert ref_key is None and out_key is None, "expected returned key to be None"

    # opcheck for torch_xcpu ops
    if dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rotary_embedding_bf16,
            (positions, query, key, head_size, rope.cos_sin_cache, is_neox_style),
        )
    elif dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rotary_embedding_fp32,
            (positions, query, key, head_size, rope.cos_sin_cache, is_neox_style),
        )
