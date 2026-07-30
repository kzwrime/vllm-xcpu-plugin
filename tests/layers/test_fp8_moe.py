from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

import vllm_xcpu_plugin.layers.fp8_moe as fp8_moe


def test_register_fp8_quantization_uses_public_api_once(monkeypatch):
    registrations = []

    def register(name):
        def decorate(config_cls):
            registrations.append((name, config_cls))
            return config_cls

        return decorate

    monkeypatch.setattr(fp8_moe, "register_quantization_config", register)
    monkeypatch.setattr(fp8_moe, "_quantization_registered", False)

    fp8_moe.register_fp8_moe_quantization()
    fp8_moe.register_fp8_moe_quantization()

    assert registrations == [("fp8", fp8_moe.XcpuFp8Config)]


def test_xcpu_fp8_config_only_replaces_mcpu_moe_method(monkeypatch):
    upstream_method = object.__new__(Fp8MoEMethod)
    replacements = []

    monkeypatch.setattr(
        Fp8Config,
        "get_quant_method",
        lambda self, layer, prefix: upstream_method,
    )

    class Replacement:
        def __init__(self, config, layer):
            replacements.append((config, layer))

    monkeypatch.setattr(fp8_moe, "XcpuFp8MoEMethod", Replacement)
    layer = object.__new__(RoutedExperts)
    config = fp8_moe.XcpuFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )

    monkeypatch.setattr(
        fp8_moe,
        "current_platform",
        SimpleNamespace(device_name="cuda"),
    )
    assert config.get_quant_method(layer, "layer") is upstream_method
    assert replacements == []

    monkeypatch.setattr(
        fp8_moe,
        "current_platform",
        SimpleNamespace(device_name="mcpu"),
    )
    replacement = config.get_quant_method(layer, "layer")
    assert isinstance(replacement, Replacement)
    assert replacements == [(config, layer)]


def test_xcpu_fp8_experts_reject_ep_eplb_and_sequence_parallel():
    config = FusedMoEParallelConfig.make_no_parallel()
    assert fp8_moe.XcpuExpertsFp8._supports_parallel_config(config)

    config.use_ep = True
    assert not fp8_moe.XcpuExpertsFp8._supports_parallel_config(config)
    config.use_ep = False

    config.enable_eplb = True
    assert not fp8_moe.XcpuExpertsFp8._supports_parallel_config(config)
    config.enable_eplb = False

    config.sp_size = 2
    assert not fp8_moe.XcpuExpertsFp8._supports_parallel_config(config)


def test_setup_kernel_binds_prepack_and_runtime_to_same_backend(monkeypatch):
    method = object.__new__(fp8_moe.XcpuFp8MoEMethod)
    method.moe = SimpleNamespace()
    method.moe_quant_config = None
    method.moe_kernel = None
    method.weight_scale_name = "weight_scale_inv"
    method.weight_block_size = [128, 128]

    w13 = torch.zeros((2, 256, 128), dtype=torch.float8_e4m3fn)
    w2 = torch.zeros((2, 128, 128), dtype=torch.float8_e4m3fn)
    w13_scale = torch.ones((2, 2, 1), dtype=torch.float32)
    w2_scale = torch.ones((2, 1, 1), dtype=torch.float32)
    layer = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale_inv=w13_scale,
        w2_weight_scale_inv=w2_scale,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=None,
        _expert_routing_tables=lambda: None,
    )

    prepack_backends = []

    def prepack(weight, backend):
        prepack_backends.append(backend)
        return weight.clone()

    def replace(layer, name, value):
        setattr(layer, name, value)

    prepare_finalize = SimpleNamespace(
        activation_format=fp8_moe.mk.FusedMoEActivationFormat.Standard
    )
    kernels = []

    class Kernel:
        def __init__(self, prepare, experts):
            self.prepare = prepare
            self.experts = experts
            kernels.append(self)

    import torch_xcpu

    monkeypatch.setattr(
        torch_xcpu.ops,
        "resolve_fp8_moe_backend",
        lambda: "acc_fp8_backend",
    )
    monkeypatch.setattr(fp8_moe, "_prepack_expert_weight", prepack)
    monkeypatch.setattr(fp8_moe, "replace_parameter", replace)
    monkeypatch.setattr(
        fp8_moe,
        "maybe_make_prepare_finalize",
        lambda **kwargs: prepare_finalize,
    )
    monkeypatch.setattr(fp8_moe.mk, "FusedMoEKernel", Kernel)

    method._setup_kernel(
        layer,
        w13,
        w2,
        w13_scale,
        w2_scale,
        None,
        None,
    )

    assert prepack_backends == ["acc_fp8_backend", "acc_fp8_backend"]
    assert len(kernels) == 1
    assert kernels[0].experts.backend == "acc_fp8_backend"
    assert kernels[0].experts.scale_block_size == (128, 128)
    assert method.moe_kernel is kernels[0]


def test_xcpu_fp8_experts_pass_weight_block_size_to_kernel(monkeypatch):
    w13_scale = torch.ones((2, 2, 1), dtype=torch.float32)
    w2_scale = torch.ones((2, 1, 1), dtype=torch.float32)
    quant_config = fp8_w8a16_moe_quant_config(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )
    assert quant_config.block_shape is None

    experts = object.__new__(fp8_moe.XcpuExpertsFp8)
    experts.moe_config = SimpleNamespace(
        experts_per_token=1,
        routing_method=RoutingMethodType.Default,
    )
    experts.quant_config = quant_config
    experts.backend = "acc_fp8_backend"
    experts.scale_block_size = (128, 128)

    captured = {}

    def run_fp8_moe(
        hidden_states,
        packed_w13,
        packed_w2,
        actual_w13_scale,
        actual_w2_scale,
        topk_weights,
        topk_ids,
        block_shape,
        backend,
    ):
        captured["w13_scale"] = actual_w13_scale
        captured["w2_scale"] = actual_w2_scale
        captured["block_shape"] = block_shape
        captured["backend"] = backend
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(fp8_moe, "_run_fp8_moe", run_fp8_moe)

    hidden_states = torch.zeros((1, 128), dtype=torch.bfloat16)
    output = experts.apply(
        hidden_states=hidden_states,
        w1=torch.empty((2, 256, 128), dtype=torch.float8_e4m3fn),
        w2=torch.empty((2, 128, 128), dtype=torch.float8_e4m3fn),
        router_logits=torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        activation=MoEActivation.SILU,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=False,
    )

    assert output.shape == hidden_states.shape
    assert captured["w13_scale"] is w13_scale
    assert captured["w2_scale"] is w2_scale
    assert captured["block_shape"] == [128, 128]
    assert captured["backend"] == "acc_fp8_backend"
