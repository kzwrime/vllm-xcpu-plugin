# SPDX-License-Identifier: Apache-2.0
"""XCPU OOT implementations for Conv2dLayer and Conv3dLayer."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer


@Conv2dLayer.register_oot
class XcpuConv2dLayer(Conv2dLayer):
    """Route patch-embedding Conv2d mulmat through torch_xcpu.ops.linear."""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # Reference: vLLM
        # vllm/model_executor/layers/conv.py::Conv2dLayer.forward_native
        # vllm/model_executor/layers/conv.py::Conv2dLayer._forward_mulmat
        assert x.dim() == 4
        if not self.enable_linear or any(d != 1 for d in self.dilation):
            return self.forward_native(x)

        import torch_xcpu

        b, _, h, w = x.shape
        k1, k2 = self.kernel_size
        h, w = h // k1, w // k2
        x = x.unfold(2, k1, k1).unfold(3, k2, k2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.input_size)
        x = torch_xcpu.ops.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        return x.view(b, h, w, self.out_channels).permute(0, 3, 1, 2)


@Conv3dLayer.register_oot
class XcpuConv3dLayer(Conv3dLayer):
    """Route patch-embedding Conv3d mulmat through torch_xcpu.ops.linear."""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # Reference: vLLM
        # vllm/model_executor/layers/conv.py::Conv3dLayer.forward_native
        # vllm/model_executor/layers/conv.py::Conv3dLayer._forward_mulmat
        assert x.dim() == 5
        if not self.enable_linear or any(d != 1 for d in self.dilation):
            return self.forward_native(x)

        import torch_xcpu

        b, _, t, h, w = x.shape
        k1, k2, k3 = self.kernel_size
        t, h, w = t // k1, h // k2, w // k3
        x = x.unfold(2, k1, k1).unfold(3, k2, k2).unfold(4, k3, k3)
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.input_size)
        x = torch_xcpu.ops.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        return x.view(b, t, h, w, self.out_channels).permute(0, 4, 1, 2, 3)
