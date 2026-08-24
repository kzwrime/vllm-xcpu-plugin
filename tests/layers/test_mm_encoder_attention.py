# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_in_xcpu_process(code: str) -> dict:
    repo = Path(__file__).parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["VLLM_PLUGINS"] = "xcpu_platform_plugin"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_mm_encoder_attention_matches_tp1_cpu_reference():
    code = r"""
import json

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_xcpu_plugin.fake_triton.runtime import get_registry
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
import vllm_xcpu_plugin.layers.mm_encoder_attention  # noqa: F401


def cpu_reference(q, k, v, scale):
    output = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        dropout_p=0.0,
        scale=scale,
    )
    return output.permute(0, 2, 1, 3).contiguous()


def ragged_cpu_reference(q, k, v, lengths, scale):
    outputs = []
    start = 0
    for length in lengths:
        end = start + length
        outputs.append(
            cpu_reference(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                scale,
            )
        )
        start = end
    return torch.cat(outputs, dim=1)


def run_mcpu(module, q, k, v, cu_seqlens=None):
    output = module(
        q.to("mcpu"),
        k.to("mcpu"),
        v.to("mcpu"),
        cu_seqlens=(
            None if cu_seqlens is None else cu_seqlens.to("mcpu")
        ),
    )
    assert output.device.type == "mcpu"
    assert output.dtype == torch.bfloat16
    torch.mcpu.synchronize()
    return output.cpu()


def make_qkv(seq_len, seed, batch_size=1):
    torch.manual_seed(seed)
    shape = (batch_size, seq_len, 12, 64)
    head_offsets = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1)
    query = (torch.randn(shape) * 0.2 + head_offsets * 0.03).bfloat16()
    key = (torch.randn(shape) * 0.2 - head_offsets * 0.02).bfloat16()
    value = (torch.randn(shape) * 0.2 + head_offsets * 0.05).bfloat16()
    return query, key, value


def error_metrics(actual, reference):
    difference = (actual.float() - reference.float()).abs()
    relative = difference / reference.float().abs().clamp_min(1e-2)
    return {
        "max_abs": float(difference.max()),
        "max_rel_floor_atol": float(relative.max()),
    }


torch.set_default_dtype(torch.bfloat16)
register_vllm_kernels()
vllm_config = VllmConfig()
assert vllm_config.parallel_config.tensor_parallel_size == 1

with set_current_vllm_config(vllm_config):
    attention = MMEncoderAttention(num_heads=12, head_size=64)
    scaled_attention = MMEncoderAttention(
        num_heads=12,
        head_size=64,
        scale=1.0,
    )

assert attention.attn_backend == AttentionBackendEnum.TORCH_SDPA
assert attention.scale == 0.125

with torch.inference_mode():
    scale_q, scale_k, scale_v = make_qkv(19, 40)
    scaled_actual = run_mcpu(
        scaled_attention,
        scale_q,
        scale_k,
        scale_v,
    )
    scaled_reference = cpu_reference(
        scale_q,
        scale_k,
        scale_v,
        scale=1.0,
    )
    default_scale_reference = cpu_reference(
        scale_q,
        scale_k,
        scale_v,
        scale=0.125,
    )
    torch.testing.assert_close(
        scaled_actual,
        scaled_reference,
        atol=1e-2,
        rtol=1e-2,
    )
    assert (scaled_actual.float() - default_scale_reference.float()).abs().max() > 1e-2

    fixed_q, fixed_k, fixed_v = make_qkv(288, 41, batch_size=2)
    fixed_actual = run_mcpu(attention, fixed_q, fixed_k, fixed_v)
    fixed_reference = cpu_reference(
        fixed_q,
        fixed_k,
        fixed_v,
        attention.scale,
    )
    assert fixed_actual.shape == (2, 288, 12, 64)
    assert torch.isfinite(fixed_actual.float()).all()
    assert torch.count_nonzero(fixed_actual) > 0
    torch.testing.assert_close(
        fixed_actual,
        fixed_reference,
        atol=1e-2,
        rtol=1e-2,
    )

    lengths = [280, 280, 280]
    cu_seqlens = torch.tensor([0, 280, 560, 840], dtype=torch.int32)
    ragged_q, ragged_k, ragged_v = make_qkv(840, 42)
    ragged_actual = run_mcpu(
        attention,
        ragged_q,
        ragged_k,
        ragged_v,
        cu_seqlens,
    )
    ragged_reference = ragged_cpu_reference(
        ragged_q,
        ragged_k,
        ragged_v,
        lengths,
        attention.scale,
    )
    assert ragged_actual.shape == (1, 840, 12, 64)
    torch.testing.assert_close(
        ragged_actual,
        ragged_reference,
        atol=1e-2,
        rtol=1e-2,
    )

    changed_v = ragged_v.clone()
    changed_v[:, 560:] += 1.0
    changed_actual = run_mcpu(
        attention,
        ragged_q,
        ragged_k,
        changed_v,
        cu_seqlens,
    )
    torch.testing.assert_close(
        changed_actual[:, :560],
        ragged_actual[:, :560],
        atol=0.0,
        rtol=0.0,
    )
    changed_tail_difference = (
        changed_actual[:, 560:].float() - ragged_actual[:, 560:].float()
    ).abs()
    assert changed_tail_difference.max() > 0.5

triton_qualname = (
    "vllm.v1.attention.ops.triton_prefill_attention._fwd_kernel"
)
print(json.dumps({
    "backend": attention.attn_backend.name,
    "implementation": type(attention).__name__,
    "global_sdpa_registered": torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "PrivateUse1",
    ),
    "dense_xcpu_sdpa_registered": (
        "torch_xcpu::scaled_dot_product_attention_out"
        in torch._C._dispatch_get_all_op_names()
    ),
    "varlen_xcpu_sdpa_registered": (
        "torch_xcpu::scaled_dot_product_attention_varlen_out"
        in torch._C._dispatch_get_all_op_names()
    ),
    "tp_world_size": vllm_config.parallel_config.tensor_parallel_size,
    "fixed_shape": list(fixed_actual.shape),
    "fixed": error_metrics(fixed_actual, fixed_reference),
    "ragged_shape": list(ragged_actual.shape),
    "ragged_cu_seqlens": cu_seqlens.tolist(),
    "ragged": error_metrics(ragged_actual, ragged_reference),
    "triton_fwd_launches": get_registry().launch_counts().get(
        triton_qualname,
        0,
    ),
}))
"""
    payload = _run_in_xcpu_process(code)

    assert payload["backend"] == "TORCH_SDPA"
    assert payload["implementation"] == "XcpuMMEncoderAttention"
    assert not payload["global_sdpa_registered"]
    assert not payload["dense_xcpu_sdpa_registered"]
    assert payload["varlen_xcpu_sdpa_registered"]
    assert payload["tp_world_size"] == 1
    assert payload["fixed_shape"] == [2, 288, 12, 64]
    assert payload["ragged_shape"] == [1, 840, 12, 64]
    assert payload["ragged_cu_seqlens"] == [0, 280, 560, 840]
    assert payload["triton_fwd_launches"] == 0
    assert payload["fixed"]["max_abs"] <= 1e-2
    assert payload["fixed"]["max_rel_floor_atol"] <= 2e-2
    assert payload["ragged"]["max_abs"] <= 1e-2
    assert payload["ragged"]["max_rel_floor_atol"] <= 2e-2
