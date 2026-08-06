from types import SimpleNamespace

import torch

from vllm_xcpu_plugin.layers.mxfp4_linear import (
    XcpuMxFp4LinearKernel,
)


def test_process_weights_uses_initialized_torch_xcpu_operator(monkeypatch):
    import torch_xcpu

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.zeros((32, 64), dtype=torch.uint8),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "weight_scale",
                torch.nn.Parameter(
                    torch.ones((32, 4), dtype=torch.uint8),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.ones(32, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )

    calls = []

    class FakeMxfp4Linear:
        def __init__(self, weight, weight_scale, bias):
            self.params = SimpleNamespace(
                packed_weight=weight.clone(),
                packed_weight_scale=weight_scale.clone(),
                bias=bias,
                backend=SimpleNamespace(name="INTEL_AMX"),
            )

        def __call__(self, x):
            return x

    def initialize(weight, weight_scale, bias):
        calls.append((weight, weight_scale, bias))
        return FakeMxfp4Linear(weight, weight_scale, bias)

    monkeypatch.setattr(torch_xcpu.ops, "initialize_mxfp4_linear", initialize)
    kernel = object.__new__(XcpuMxFp4LinearKernel)
    layer = Layer()
    kernel.process_weights_after_loading(layer)

    assert len(calls) == 1
    assert layer.bias.dtype == torch.float32
    assert calls[0][2].data_ptr() == layer.bias.data_ptr()
    assert (
        layer.weight.data_ptr()
        == layer._xcpu_mxfp4_linear.params.packed_weight.data_ptr()
    )
    assert (
        layer.weight_scale.data_ptr()
        == layer._xcpu_mxfp4_linear.params.packed_weight_scale.data_ptr()
    )

    x = torch.ones((2, 128), dtype=torch.bfloat16)
    assert kernel.apply_weights(layer, x, layer.bias) is x
