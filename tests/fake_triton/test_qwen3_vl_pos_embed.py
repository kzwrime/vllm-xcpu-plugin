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
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_bilinear_pos_embed_matches_native_reference():
    code = r"""
import json
import torch
import torch_mcpu

from vllm_xcpu_plugin.fake_triton import install_fake_triton

install_fake_triton(replace_existing=True)

from vllm.model_executor.models.qwen3_vl import (
    pos_embed_interpolate_native,
    triton_pos_embed_interpolate,
)
from vllm_xcpu_plugin.fake_triton.runtime import get_registry
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()

cases = [
    (torch.float32, 1, 2, 2),
    (torch.bfloat16, 1, 2, 2),
    (torch.float32, 1, 8, 36),
    (torch.bfloat16, 1, 8, 36),
    (torch.float32, 3, 14, 20),
    (torch.bfloat16, 3, 14, 20),
    (torch.bfloat16, 1, 256, 256),
]
tolerances = {
    torch.float32: (5e-5, 1e-5),
    torch.bfloat16: (1e-2, 1e-2),
}
metrics = []

for dtype, t, h, w in cases:
    torch.manual_seed(42)
    embed_cpu = torch.randn((48 * 48, 768), dtype=dtype) * 0.25
    reference = pos_embed_interpolate_native(
        embed_cpu, t, h, w, 48, 2, dtype
    )
    actual = triton_pos_embed_interpolate(
        embed_cpu.to("mcpu"), t, h, w, 48, 2, dtype
    ).cpu()

    assert actual.shape == (t * h * w, 768)
    assert actual.dtype == dtype
    atol, rtol = tolerances[dtype]
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    difference = (actual.float() - reference.float()).abs()
    relative = difference / reference.float().abs().clamp_min(atol)
    metrics.append({
        "dtype": str(dtype).removeprefix("torch."),
        "grid_thw": [t, h, w],
        "shape": list(actual.shape),
        "max_abs": float(difference.max()),
        "max_rel_floor_atol": float(relative.max()),
    })

qualname = "vllm.model_executor.models.qwen3_vl._bilinear_pos_embed_kernel"
print(json.dumps({
    "metrics": metrics,
    "launches": get_registry().launch_counts()[qualname],
}))
"""
    payload = _run_in_xcpu_process(code)

    assert payload["launches"] == 7
    assert [metric["grid_thw"] for metric in payload["metrics"]] == [
        [1, 2, 2],
        [1, 2, 2],
        [1, 8, 36],
        [1, 8, 36],
        [3, 14, 20],
        [3, 14, 20],
        [1, 256, 256],
    ]


def test_bilinear_pos_embed_rejects_invalid_launches():
    code = r"""
import json
import torch
import torch_mcpu

from vllm_xcpu_plugin.fake_triton import install_fake_triton

install_fake_triton(replace_existing=True)

from vllm.model_executor.models.qwen3_vl import (
    _bilinear_pos_embed_kernel,
    triton_pos_embed_interpolate,
)
from vllm_xcpu_plugin.fake_triton.runtime import InvalidLaunchError
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()
embed = torch.randn((48 * 48, 8), dtype=torch.bfloat16, device="mcpu")

def rejects(call, error_type):
    try:
        call()
    except error_type:
        return True
    return False

non_divisible = rejects(
    lambda: triton_pos_embed_interpolate(
        embed, 1, 3, 4, 48, 2, torch.bfloat16
    ),
    AssertionError,
)
bad_grid = rejects(
    lambda: _bilinear_pos_embed_kernel[(7,)](
        embed,
        torch.empty((7, 8), dtype=torch.bfloat16, device="mcpu"),
        2,
        2,
        47.0,
        47.0,
        48,
        2,
        8,
        8,
    ),
    InvalidLaunchError,
)
bad_block = rejects(
    lambda: _bilinear_pos_embed_kernel[(4,)](
        embed,
        torch.empty((4, 8), dtype=torch.bfloat16, device="mcpu"),
        2,
        2,
        47.0,
        47.0,
        48,
        2,
        8,
        4,
    ),
    InvalidLaunchError,
)
bad_scale = rejects(
    lambda: _bilinear_pos_embed_kernel[(4,)](
        embed,
        torch.empty((4, 8), dtype=torch.bfloat16, device="mcpu"),
        2,
        2,
        1.0,
        47.0,
        48,
        2,
        8,
        8,
    ),
    InvalidLaunchError,
)
embed_fp16 = embed.to(torch.float16)
bad_dtype = rejects(
    lambda: triton_pos_embed_interpolate(
        embed_fp16, 1, 2, 2, 48, 2, torch.float16
    ),
    InvalidLaunchError,
)
print(json.dumps({
    "non_divisible": non_divisible,
    "bad_grid": bad_grid,
    "bad_block": bad_block,
    "bad_scale": bad_scale,
    "bad_dtype": bad_dtype,
}))
"""
    payload = _run_in_xcpu_process(code)

    assert payload == {
        "non_divisible": True,
        "bad_grid": True,
        "bad_block": True,
        "bad_scale": True,
        "bad_dtype": True,
    }
