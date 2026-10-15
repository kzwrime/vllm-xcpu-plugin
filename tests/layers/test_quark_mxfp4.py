from types import SimpleNamespace

import pytest
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
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

    scheme = XcpuQuarkOCPMXLinearScheme(kMxfp4Static, None)

    assert scheme.activation_quant_key is None
    assert scheme.input_dtype is None
    assert scheme._use_xcpu_w4a16


def test_mla_mxfp4_preserves_nibble_order_sign_and_e8m0_scale():
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

    layer = torch.nn.Module()
    layer._xcpu_mla_weight_only = True
    layer.params_dtype = torch.bfloat16
    # Low nibble precedes high nibble in each packed byte.
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])
    layer.weight = torch.nn.Parameter(
        packed.to(torch.uint8).repeat(2).view(1, 16), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[128]], dtype=torch.uint8), requires_grad=False
    )
    scheme = XcpuQuarkOCPMXLinearScheme(kMxfp4Static, None)

    scheme.process_weights_after_loading(layer)

    expected = (
        torch
        .tensor(
            [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
            dtype=torch.bfloat16,
        )
        .repeat(2)
        .view(1, 32)
    )
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    assert not hasattr(layer, "_xcpu_quark_mxfp4_linear")


def test_native_w4a16_rebuilds_quark_moe(monkeypatch):
    upstream = object.__new__(QuarkOCP_MX_MoEMethod)
    upstream.weight_dtype = "mxfp4"
    upstream.input_dtype = None
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

    upstream.weight_quant_key = kMxfp4Static
    upstream.activation_quant_key = None
    moe_config = object()
    layer = SimpleNamespace(moe_config=moe_config)
    calls = []

    def fake_method(moe, weight_quant_key, activation_quant_key):
        calls.append((moe, weight_quant_key, activation_quant_key))
        return "xcpu-w4a16-moe"

    monkeypatch.setattr(quark_mxfp4, "XcpuQuarkOCPMXMoEMethod", fake_method)

    result = xcpu_moe_method_factory(upstream, None, layer)

    assert result == "xcpu-w4a16-moe"
    assert calls == [(moe_config, upstream.weight_quant_key, None)]


def test_quark_w4a4_moe_keeps_upstream_method():
    upstream = object.__new__(QuarkOCP_MX_MoEMethod)
    upstream.weight_dtype = "mxfp4"
    upstream.input_dtype = "mxfp4"
    layer = SimpleNamespace(moe_config=object())

    assert xcpu_moe_method_factory(upstream, None, layer) is upstream


def _mxfp4_moe_config():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    return SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=False,
            enable_eplb=False,
        ),
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
        has_hash_routing=False,
        routing_method=RoutingMethodType.DeepSeekV3,
        router_logits_dtype=None,
        hidden_dim=64,
        is_lora_enabled=False,
    )


def test_platform_selects_xcpu_mxfp4_moe_experts(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    from vllm_xcpu_plugin.layers.fused_moe.grouped_gemm_experts import (
        XcpuGroupedGemmExperts,
    )
    from vllm_xcpu_plugin.platform import McpuPlatform

    config = _mxfp4_moe_config()
    expected = (Mxfp4MoeBackend.CPU, XcpuGroupedGemmExperts)
    assert McpuPlatform.select_mxfp4_moe_backend(config, None) == expected

    monkeypatch.setattr(quark_mxfp4, "force_quark_mxfp4_w4a16", lambda: False)
    with pytest.raises(NotImplementedError, match="quantization scheme"):
        McpuPlatform.select_mxfp4_moe_backend(config, kMxfp4Dynamic)

    monkeypatch.setattr(quark_mxfp4, "force_quark_mxfp4_w4a16", lambda: True)
    assert McpuPlatform.select_mxfp4_moe_backend(config, kMxfp4Dynamic) == expected


def test_quark_moe_bridge_rejects_mxfp4_activations():
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
        kMxfp4Static,
    )

    with pytest.raises(ValueError, match="only supports W4A16"):
        XcpuQuarkOCPMXMoEMethod(object(), kMxfp4Static, kMxfp4Dynamic)
