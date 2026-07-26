from types import SimpleNamespace

import torch
from vllm.model_executor.kernels import linear
from vllm.platforms.interface import PlatformEnum

from vllm_xcpu_plugin.layers.fp8_linear import (
    XcpuFp8BlockScaledMMLinearKernel,
    register_fp8_linear_kernel,
)


def test_register_fp8_linear_kernel_for_oot_platform_is_idempotent():
    register_fp8_linear_kernel()
    register_fp8_linear_kernel()

    kernels = linear._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT]
    assert kernels[0] is XcpuFp8BlockScaledMMLinearKernel
    assert kernels.count(XcpuFp8BlockScaledMMLinearKernel) == 1


def test_register_fp8_linear_kernel_prefers_public_api(monkeypatch):
    calls = []

    def record_registration(kernel_class, platform, kernel_type):
        calls.append((kernel_class, platform, kernel_type))

    monkeypatch.setattr(linear, "register_linear_kernel", record_registration)
    before = list(linear._POSSIBLE_FP8_BLOCK_KERNELS.get(PlatformEnum.OOT, []))

    register_fp8_linear_kernel()

    assert calls == [
        (
            XcpuFp8BlockScaledMMLinearKernel,
            PlatformEnum.OOT,
            "fp8_block",
        )
    ]
    assert linear._POSSIBLE_FP8_BLOCK_KERNELS.get(PlatformEnum.OOT, []) == before


def test_process_weights_uses_initialized_torch_xcpu_operator(monkeypatch):
    import torch_xcpu

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "weight_scale_inv",
                torch.nn.Parameter(
                    torch.ones((1, 1), dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.ones(128, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )
            self.weight_scale = None

    calls = []

    class FakeFp8Linear:
        def __init__(self, weight, weight_scale, bias):
            self.params = SimpleNamespace(
                packed_weight=weight.clone(),
                weight_scale=weight_scale,
                bias=bias,
            )

        def __call__(self, x):
            return x

    def initialize(weight, weight_scale, block_size, bias):
        calls.append((weight, weight_scale, block_size, bias))
        return FakeFp8Linear(weight, weight_scale, bias)

    monkeypatch.setattr(torch_xcpu.ops, "initialize_fp8_linear", initialize)

    kernel = object.__new__(XcpuFp8BlockScaledMMLinearKernel)
    kernel.weight_group_shape = (128, 128)
    layer = Layer()
    kernel.process_weights_after_loading(layer)

    assert len(calls) == 1
    assert calls[0][2] == (128, 128)
    assert calls[0][3].dtype == torch.float32
    assert (
        layer.weight.data_ptr()
        == layer._xcpu_fp8_linear.params.packed_weight.data_ptr()
    )
    assert (
        layer.weight_scale_inv.data_ptr()
        == layer._xcpu_fp8_linear.params.weight_scale.data_ptr()
    )

    x = torch.ones((2, 128), dtype=torch.bfloat16)
    assert kernel.apply_weights(layer, x, layer.bias) is x
