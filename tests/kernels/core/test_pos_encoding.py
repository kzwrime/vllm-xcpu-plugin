# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass

import pytest
import torch
import torch_xcpu  # noqa: F401
from test_activation import _model_filter_matches
from torch_xcpu.model_configs import ALL_MODEL_CONFIGS, COMMON_TOKENS
from vllm.model_executor.layers.rotary_embedding import get_rope
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
)  # 从项目导入 opcheck

load_general_plugins()


# =============================================================================
# Test Configuration
# =============================================================================
IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float]
# Modified to cover Qwen2/Qwen3 (128 is typical for 7B/72B, 64 for 0.5B)
HEAD_SIZES = [64, 80]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [1, 16]
BATCH_SIZES = [1, 5]
SEQ_LENS = [1, 2, 7, 11, 55, 333, 1333]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = CUSTOM_OP_TEST_DEVICES
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


@torch.inference_mode()
def _test_rotary_embedding_model(
    rope,
    rope_fp32,
    query_cpu: torch.Tensor,
    key_cpu: torch.Tensor | None,
    positions_cpu: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    positions: torch.Tensor,
    is_neox_style: bool,
    head_size: int,
    dtype: torch.dtype,
    use_key: bool,
) -> None:
    # Compute reference in fp32 for higher precision
    query_fp32 = query_cpu.to(torch.float32)
    key_fp32 = key_cpu.to(torch.float32) if key_cpu is not None else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope_fp32.forward_native(
        positions_cpu,
        query_fp32.clone(),
        key_fp32.clone() if key_fp32 is not None else None,
    )
    out_query, out_key = rope.forward(
        positions, query.view(query.shape[0], -1), key.view(key.shape[0], -1)
    )
    torch.accelerator.synchronize()
    out_query_cpu = out_query.view(ref_query.shape).cpu()
    out_key_cpu = out_key.view(ref_key.shape).cpu() if out_key is not None else None

    # Compare using both assert_close and default_dice_tol
    # Reference precision is fp32, tolerance based on target (out_query) dtype
    atol = get_default_atol(out_query_cpu)
    rtol = get_default_rtol(out_query_cpu)
    torch.testing.assert_close(
        out_query_cpu.to(torch.float32),
        ref_query,
        atol=atol,
        rtol=rtol,
    )
    if use_key:
        assert out_key_cpu is not None
        torch.testing.assert_close(
            out_key_cpu.to(torch.float32),
            ref_key,
            atol=atol,
            rtol=rtol,
        )
    else:
        assert ref_key is None and out_key is None, "expected returned key to be None"

    # Check Dice tolerance
    diff_query = calc_diff(out_query_cpu.to(torch.float32), ref_query)
    assert diff_query < default_dice_tol, (
        f"Query diff {diff_query} exceeds dice tolerance {default_dice_tol}"
    )
    if use_key:
        assert out_key_cpu is not None
        diff_key = calc_diff(out_key_cpu.to(torch.float32), ref_key)
        assert diff_key < default_dice_tol, (
            f"Key diff {diff_key} exceeds dice tolerance {default_dice_tol}"
        )

    # opcheck for torch_xcpu ops
    if dtype == torch.bfloat16:
        opcheck(
            torch.ops.torch_xcpu.rotary_embedding_bf16,
            (positions, query, key, head_size, rope.cos_sin_cache, is_neox_style),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )
    elif dtype == torch.float:
        opcheck(
            torch.ops.torch_xcpu.rotary_embedding_fp32,
            (positions, query, key, head_size, rope.cos_sin_cache, is_neox_style),
            cond=CUSTOM_OP_TEST_ENABLE_OPCHECK,
        )


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_key", USE_KEY)
@torch.inference_mode()
def _test_rotary_embedding_basic(
    default_vllm_config,
    is_neox_style: bool,
    head_size: int,
    rotary_dim: int | None,
    dtype: torch.dtype,
    seed: int,
    device: str,
    use_key: bool,
    subtests,
    max_position: int = 8192,
    rope_theta: float = 10000,
) -> None:

    if rotary_dim is None:
        rotary_dim = head_size

    set_random_seed(seed)
    if rotary_dim is None:
        rotary_dim = head_size
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": rotary_dim / head_size,
    }
    rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
    rope_fp32 = rope.to(dtype=torch.float32)
    rope = rope.to(dtype=dtype, device=device)

    for tensor_shape_fn in TENSORS_SHAPES_FN:
        for batch_size in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                for num_heads in NUM_HEADS:
                    with subtests.test(
                        msg="rotary_embedding_config",
                        tensor_shape_fn=tensor_shape_fn.__name__,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_heads=num_heads,
                    ):
                        positions_cpu = torch.randint(
                            0, max_position, (batch_size, seq_len), device="cpu"
                        )
                        query_shape = tensor_shape_fn(
                            batch_size, seq_len, num_heads, head_size
                        )
                        query_base_cpu = torch.randn(
                            query_shape, dtype=dtype, device="cpu"
                        )

                        # slice tensor if required, noop otherwise
                        query_cpu = query_base_cpu[..., :head_size]
                        query = query_base_cpu.to(device)[..., :head_size]

                        if use_key:
                            key_base_cpu = torch.randn(
                                query_shape, dtype=dtype, device="cpu"
                            )
                            key_cpu = key_base_cpu[..., :head_size]
                            key = key_base_cpu.to(device)[..., :head_size]
                        else:
                            key_cpu = None
                            key = None

                        _test_rotary_embedding_model(
                            rope,
                            rope_fp32,
                            query_cpu,
                            key_cpu,
                            positions_cpu,
                            query,
                            key,
                            positions_cpu.to(device),
                            is_neox_style,
                            head_size,
                            dtype,
                            use_key,
                        )


@dataclass(frozen=True)
class RotaryEmbeddingConfig:
    """Configuration for rotary embedding test cases (automatically hashable & dedupable)."""  # noqa: E501

    num_heads: int
    num_kv_heads: int
    head_size: int
    rotary_dim: int
    rope_theta: float
    max_position: int
    contiguous_qkv: bool
    model_name: str


COMBINATIONS: set[RotaryEmbeddingConfig] = set()

for model_name, config in ALL_MODEL_CONFIGS.items():
    if not _model_filter_matches(model_name):
        continue
    if config.num_heads is None or config.head_size is None:
        continue

    num_kv_heads = (
        config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
    )
    rope_theta = config.rope_theta if config.rope_theta is not None else 10000.0
    max_position = (
        config.max_position_embeddings
        if config.max_position_embeddings is not None
        else 8192
    )
    rotary_dim = (
        config.rotary_dim if config.rotary_dim is not None else config.head_size
    )

    # Iterate over all TP sizes for this model
    for tp_size in config.tp_sizes:
        # Calculate per-rank head counts (TP splitting)
        num_heads_per_rank = config.num_heads // tp_size
        # For GQA: when num_kv_heads < tp_size, each rank gets 1 KV head
        # Otherwise, divide evenly
        num_kv_heads_per_rank = 1 if num_kv_heads < tp_size else num_kv_heads // tp_size

        COMBINATIONS.add(
            RotaryEmbeddingConfig(
                num_heads=num_heads_per_rank,
                num_kv_heads=num_kv_heads_per_rank,
                head_size=config.head_size,
                rotary_dim=rotary_dim,
                rope_theta=rope_theta,
                max_position=max_position,
                contiguous_qkv=config.contiguous_qkv,
                model_name=config.name,
            )
        )


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_key", [True])
@pytest.mark.parametrize("combinations", COMBINATIONS)
@torch.inference_mode()
def test_rotary_embedding(
    default_vllm_config,
    is_neox_style: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    use_key: bool,
    combinations: RotaryEmbeddingConfig,
    subtests,
) -> None:

    num_heads: int = combinations.num_heads
    num_kv_heads: int = combinations.num_kv_heads
    head_size: int = combinations.head_size
    rotary_dim: int = combinations.rotary_dim
    rope_theta: float = combinations.rope_theta
    max_position: int = combinations.max_position
    contiguous_qkv: bool = combinations.contiguous_qkv
    _ = combinations.model_name  # noqa: F841

    if rotary_dim is None:
        rotary_dim = head_size

    set_random_seed(seed)
    if rotary_dim is None:
        rotary_dim = head_size
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": rotary_dim / head_size,
    }
    rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
    rope_fp32 = rope.to(dtype=torch.float32)
    rope = rope.to(dtype=dtype, device=device)

    for seq_len in COMMON_TOKENS:
        with subtests.test(
            msg="rotary_embedding_config",
            seq_len=seq_len,
        ):
            if contiguous_qkv:
                # Models using .split() produce contiguous tensors
                query_cpu = torch.randn(
                    seq_len, num_heads, head_size, dtype=dtype, device="cpu"
                )
                key_cpu = torch.randn(
                    seq_len, num_kv_heads, head_size, dtype=dtype, device="cpu"
                )
            else:
                # Models using .chunk() or with TP sharding produce non-contiguous views
                # Simulate by creating a combined QKV tensor and slicing
                q_proj_size = num_heads
                k_proj_size = num_kv_heads
                v_proj_size = num_kv_heads
                total_qkv_size = q_proj_size + k_proj_size + v_proj_size

                qkv = torch.randn(
                    seq_len, total_qkv_size, head_size, dtype=dtype, device="cpu"
                )
                q_end = q_proj_size
                k_end = q_proj_size + k_proj_size
                query_cpu = qkv[:, :q_end, :]
                key_cpu = qkv[:, q_end:k_end, :]
                qkv_device = qkv.to(device)
                query = qkv_device[:, :q_end, :]
                key = qkv_device[:, q_end:k_end, :]
                assert query.stride() == query_cpu.stride()
                assert key.stride() == key_cpu.stride()
            if contiguous_qkv:
                query = query_cpu.to(device)
                key = key_cpu.to(device)

            positions_cpu = torch.randint(0, max_position, (seq_len,), device="cpu")
            _test_rotary_embedding_model(
                rope,
                rope_fp32,
                query_cpu,
                key_cpu if use_key else None,
                positions_cpu,
                query,
                key if use_key else None,
                positions_cpu.to(device),
                is_neox_style,
                head_size,
                dtype,
                use_key,
            )
