from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)

import vllm_xcpu_plugin.layers.quark_mxfp4 as quark_mxfp4
from vllm_xcpu_plugin.layers.fused_moe.routed_experts import (
    xcpu_moe_method_factory,
)
from vllm_xcpu_plugin.layers.quark_mxfp4 import (
    XcpuQuarkOCPMXLinearScheme,
    XcpuQuarkOCPMXMoEMethod,
    is_quark_mxfp4_w4a16,
)


def test_quark_w4a16_linear_uses_torch_xcpu(monkeypatch):
    import torch_xcpu

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros((32, 64), dtype=torch.uint8), requires_grad=False
            )
            self.weight_scale = torch.nn.Parameter(
                torch.ones((32, 4), dtype=torch.uint8), requires_grad=False
            )

    class FakeLinear:
        def __init__(self, weight, scale):
            self.params = SimpleNamespace(
                packed_weight=weight.clone(),
                packed_weight_scale=scale.clone(),
                bias=None,
                backend=SimpleNamespace(name="PORTABLE_LUT"),
            )

        def __call__(self, x):
            return x + 1

    calls = []

    def initialize(weight, scale, bias):
        calls.append((weight, scale, bias))
        return FakeLinear(weight, scale)

    monkeypatch.setattr(torch_xcpu.ops, "initialize_mxfp4_linear", initialize)
    monkeypatch.setattr(quark_mxfp4, "_initialize_dummy_e8m0_scale", lambda scale: None)
    scheme = object.__new__(XcpuQuarkOCPMXLinearScheme)
    scheme._use_xcpu_w4a16 = True
    layer = Layer()

    scheme.process_weights_after_loading(layer)
    result = scheme.apply_weights(layer, torch.zeros((2, 128)), None)

    assert len(calls) == 1
    packed_weight = layer._xcpu_quark_mxfp4_linear.params.packed_weight
    assert layer.weight.data_ptr() == packed_weight.data_ptr()
    assert result.eq(1).all()


def test_quark_w4a16_moe_initializes_xcpu_fused_moe(monkeypatch):
    import torch_xcpu

    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((2, 64, 32), dtype=torch.uint8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((2, 32, 32), dtype=torch.uint8), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((2, 64, 2), dtype=torch.uint8), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones((2, 32, 2), dtype=torch.uint8), requires_grad=False
    )

    method = object.__new__(XcpuQuarkOCPMXMoEMethod)
    method.moe_kernel = None
    method.has_bias = False
    method.moe = SimpleNamespace(max_num_tokens=8, experts_per_token=2)
    fused_moe = object()
    initialize_calls = []
    install_calls = []

    def initialize(*args, **kwargs):
        initialize_calls.append((args, kwargs))
        return fused_moe

    def install(*args, **kwargs):
        install_calls.append((args, kwargs))

    monkeypatch.setattr(torch_xcpu.ops, "initialize_fused_moe", initialize)
    monkeypatch.setattr(quark_mxfp4, "install_fused_moe", install)
    monkeypatch.setattr(quark_mxfp4, "_initialize_dummy_e8m0_scale", lambda scale: None)

    method.process_weights_after_loading(layer)

    assert initialize_calls[0][1] == {
        "scale_block_size": (1, 32),
        "m_capacity": 16,
    }
    initialized_tensors = initialize_calls[0][0][:4]
    expected_tensors = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
    )
    assert all(
        actual is expected
        for actual, expected in zip(initialized_tensors, expected_tensors)
    )
    assert install_calls[0][0][:3] == (method, layer, fused_moe)


def test_quark_linear_registration_replaces_factory_symbol(monkeypatch):
    import vllm.model_executor.layers.quantization.quark.quark as quark

    original = quark.QuarkOCP_MX
    monkeypatch.setattr(quark, "QuarkOCP_MX", original)

    quark_mxfp4.register_quark_mxfp4_linear_scheme()

    assert quark.QuarkOCP_MX is XcpuQuarkOCPMXLinearScheme


def test_dummy_e8m0_scale_uses_finite_unit_value(monkeypatch):
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="dummy"))
    monkeypatch.setattr(quark_mxfp4, "get_current_vllm_config_or_none", lambda: config)
    scale = torch.tensor([0, 255], dtype=torch.uint8)

    quark_mxfp4._initialize_dummy_e8m0_scale(scale)

    assert scale.tolist() == [127, 127]


def test_real_e8m0_scale_is_not_modified(monkeypatch):
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="auto"))
    monkeypatch.setattr(quark_mxfp4, "get_current_vllm_config_or_none", lambda: config)
    scale = torch.tensor([0, 255], dtype=torch.uint8)

    quark_mxfp4._initialize_dummy_e8m0_scale(scale)

    assert scale.tolist() == [0, 255]


def test_native_w4a16_accepts_only_mxfp4_without_input_quant():
    assert is_quark_mxfp4_w4a16("mxfp4", None)
    assert not is_quark_mxfp4_w4a16("mxfp4", "mxfp4")
    assert not is_quark_mxfp4_w4a16("mxfp4", "fp8")
    assert not is_quark_mxfp4_w4a16("mxfp6_e3m2", None)


def test_native_w4a16_preserves_checkpoint_activation_spec():
    weight_spec = {"dtype": "fp4", "qscheme": "per_group"}
    scheme = XcpuQuarkOCPMXLinearScheme(weight_spec, None)

    assert scheme.input_quant_spec is None
    assert scheme.input_dtype is None
    assert scheme._use_xcpu_w4a16


def test_native_w4a16_rebuilds_quark_moe(monkeypatch):
    upstream = object.__new__(QuarkOCP_MX_MoEMethod)
    upstream.weight_dtype = "mxfp4"
    upstream.input_dtype = None
    upstream.weight_quant = {"dtype": "fp4"}
    upstream.input_quant = None
    moe_config = object()
    layer = SimpleNamespace(moe_config=moe_config)
    calls = []

    def fake_method(weight_quant, input_quant, moe):
        calls.append((weight_quant, input_quant, moe))
        return "xcpu-w4a16-moe"

    monkeypatch.setattr(quark_mxfp4, "XcpuQuarkOCPMXMoEMethod", fake_method)

    result = xcpu_moe_method_factory(upstream, None, layer)

    assert result == "xcpu-w4a16-moe"
    assert calls == [(upstream.weight_quant, None, moe_config)]


def test_quark_w4a4_moe_keeps_upstream_method():
    upstream = object.__new__(QuarkOCP_MX_MoEMethod)
    upstream.weight_dtype = "mxfp4"
    upstream.input_dtype = "mxfp4"
    layer = SimpleNamespace(moe_config=object())

    assert xcpu_moe_method_factory(upstream, None, layer) is upstream
