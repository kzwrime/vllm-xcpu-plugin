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
    env["TORCHINDUCTOR_CPP_WRAPPER"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_conv2d_and_conv3d_oot_route_through_xcpu_linear():
    code = r"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
from vllm_xcpu_plugin.models.vision_conv import register_vision_conv

FORBIDDEN_OPS = {
    "aten::linear",
    "aten::mm",
    "aten::addmm",
    "mcpu::aten::mm_out",
    "mcpu::aten::addmm_out",
}


def init_layer_params(layer, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    weight = (
        torch.randn(layer.weight.shape, generator=gen, dtype=torch.float32) * 0.05
    ).to(layer.weight.dtype)
    layer.weight.data.copy_(weight)
    bias = None
    if layer.bias is not None:
        bias = (
            torch.randn(layer.bias.shape, generator=gen, dtype=torch.float32) * 0.05
        ).to(layer.bias.dtype)
        layer.bias.data.copy_(bias)
    return weight, bias


def profile_op_counts(fn):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        out = fn()
        torch.mcpu.synchronize()
    counts = {}
    for evt in prof.events():
        counts[evt.name] = counts.get(evt.name, 0) + 1
    return out, counts


register_vision_conv()
vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))

with set_current_vllm_config(vllm_config), torch.inference_mode():
    # 1. Conv2dLayer accuracy + profiler check (bias=True and bias=False, bf16 & fp32)
    for dtype, tol in ((torch.bfloat16, 1.5e-2), (torch.float32, 1e-5)):
        for use_bias, expected_op in (
            (True, "torch_xcpu::addmm_v2"),
            (False, "torch_xcpu::mm_out"),
        ):
            conv2d = Conv2dLayer(
                in_channels=3,
                out_channels=64,
                kernel_size=(16, 16),
                stride=(16, 16),
                bias=use_bias,
                params_dtype=dtype,
            )
            assert type(conv2d).__name__ == "XcpuConv2dLayer"
            assert conv2d.enable_linear
            w_cpu, b_cpu = init_layer_params(conv2d, seed=100 + int(use_bias))
            conv2d = conv2d.to("mcpu")

            x_cpu = torch.randn(2, 3, 32, 48, dtype=dtype)
            ref = F.conv2d(x_cpu, w_cpu, b_cpu, stride=(16, 16))
            x_mcpu = x_cpu.to("mcpu")
            actual, counts = profile_op_counts(lambda: conv2d(x_mcpu))
            actual_cpu = actual.cpu()
            assert actual_cpu.shape == (2, 64, 2, 3)
            torch.testing.assert_close(actual_cpu, ref, atol=tol, rtol=tol)
            assert counts.get(expected_op, 0) == 1, counts
            for bad in FORBIDDEN_OPS:
                assert counts.get(bad, 0) == 0, (bad, counts)

    # 2. Conv3dLayer accuracy + profiler check across aligned & unaligned patch rows
    for dtype, tol in ((torch.bfloat16, 1.5e-2), (torch.float32, 1e-5)):
        for use_bias, expected_op in (
            (True, "torch_xcpu::addmm_v2"),
            (False, "torch_xcpu::mm_out"),
        ):
            conv3d = Conv3dLayer(
                in_channels=3,
                out_channels=96,
                kernel_size=(2, 16, 16),
                stride=(2, 16, 16),
                bias=use_bias,
                params_dtype=dtype,
            )
            assert type(conv3d).__name__ == "XcpuConv3dLayer"
            assert conv3d.enable_linear
            w_cpu, b_cpu = init_layer_params(conv3d, seed=200 + int(use_bias))
            conv3d = conv3d.to("mcpu")

            # Qwen3_VisionPatchEmbed single-patch-per-row shapes (unaligned & aligned L)
            for num_patches in (1, 7, 16, 288):
                x_cpu = torch.randn(num_patches, 3, 2, 16, 16, dtype=dtype)
                ref = F.conv3d(x_cpu, w_cpu, b_cpu, stride=(2, 16, 16)).view(
                    num_patches, 96
                )
                x_mcpu = x_cpu.to("mcpu")
                actual, counts = profile_op_counts(
                    lambda: conv3d(x_mcpu).view(num_patches, 96)
                )
                actual_cpu = actual.cpu()
                assert actual_cpu.shape == (num_patches, 96)
                torch.testing.assert_close(actual_cpu, ref, atol=tol, rtol=tol)
                assert counts.get(expected_op, 0) == 1, counts
                for bad in FORBIDDEN_OPS:
                    assert counts.get(bad, 0) == 0, (bad, counts)

            # Multi-window 5D input
            x_grid = torch.randn(2, 3, 4, 32, 48, dtype=dtype)
            ref_grid = F.conv3d(x_grid, w_cpu, b_cpu, stride=(2, 16, 16))
            actual_grid = conv3d(x_grid.to("mcpu")).cpu()
            assert actual_grid.shape == (2, 96, 2, 2, 3)
            torch.testing.assert_close(actual_grid, ref_grid, atol=tol, rtol=tol)

    # 3. Non-linear conv fallback (enable_linear=False) delegates to forward_native
    non_linear_2d = Conv2dLayer(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        params_dtype=torch.float32,
    )
    assert not non_linear_2d.enable_linear
    w2_cpu, b2_cpu = init_layer_params(non_linear_2d, seed=301)
    x2_cpu = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    ref2 = F.conv2d(x2_cpu, w2_cpu, b2_cpu, stride=1, padding=1)
    torch.testing.assert_close(non_linear_2d(x2_cpu), ref2, atol=1e-5, rtol=1e-5)

    non_linear_3d = Conv3dLayer(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        params_dtype=torch.float32,
    )
    assert not non_linear_3d.enable_linear
    w3_cpu, b3_cpu = init_layer_params(non_linear_3d, seed=302)
    x3_cpu = torch.randn(2, 3, 4, 6, 6, dtype=torch.float32)
    ref3 = F.conv3d(x3_cpu, w3_cpu, b3_cpu, stride=1, padding=1)
    torch.testing.assert_close(non_linear_3d(x3_cpu), ref3, atol=1e-5, rtol=1e-5)

    # 4. Dynamic-L torch.compile check matching Qwen3_VisionPatchEmbed.forward
    class PatchEmbedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Conv3dLayer(
                in_channels=3,
                out_channels=128,
                kernel_size=(2, 16, 16),
                stride=(2, 16, 16),
                bias=True,
                params_dtype=torch.bfloat16,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            num_rows = x.shape[0]
            x = x.view(num_rows, -1, 2, 16, 16)
            return self.proj(x).view(num_rows, 128)

    patch_embed = PatchEmbedModule()
    wp_cpu, bp_cpu = init_layer_params(patch_embed.proj, seed=401)
    patch_embed = patch_embed.to("mcpu")
    compiled_patch_embed = torch.compile(
        patch_embed,
        dynamic=True,
        options={
            "epilogue_fusion": False,
            "pattern_matcher": False,
            "combo_kernels": False,
            "benchmark_combo_kernel": False,
        },
    )

    compile_checked_lengths = []
    for length in (7, 16, 35):
        flat_cpu = torch.randn(length, 3 * 2 * 16 * 16, dtype=torch.bfloat16)
        ref_patch = F.conv3d(
            flat_cpu.view(length, 3, 2, 16, 16),
            wp_cpu,
            bp_cpu,
            stride=(2, 16, 16),
        ).view(length, 128)
        flat_mcpu = flat_cpu.to("mcpu")
        if length == 7:
            # First call compiles; run once more to profile steady-state compiled graph.
            _ = compiled_patch_embed(flat_mcpu)
            torch.mcpu.synchronize()
        out_patch, c_counts = profile_op_counts(lambda: compiled_patch_embed(flat_mcpu))
        torch.testing.assert_close(out_patch.cpu(), ref_patch, atol=1.5e-2, rtol=1.5e-2)
        assert c_counts.get("torch_xcpu::addmm_v2", 0) == 1, c_counts
        for bad in FORBIDDEN_OPS:
            assert c_counts.get(bad, 0) == 0, (bad, c_counts)
        compile_checked_lengths.append(length)

print(json.dumps({
    "conv2d_cls": "XcpuConv2dLayer",
    "conv3d_cls": "XcpuConv3dLayer",
    "compile_checked_lengths": compile_checked_lengths,
}))
"""
    payload = _run_in_xcpu_process(code)
    assert payload["conv2d_cls"] == "XcpuConv2dLayer"
    assert payload["conv3d_cls"] == "XcpuConv3dLayer"
    assert payload["compile_checked_lengths"] == [7, 16, 35]
