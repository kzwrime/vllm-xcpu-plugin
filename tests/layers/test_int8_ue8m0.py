# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_xcpu_plugin xcpu_int8 (Int8 + UE8M0) integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoEParallelConfig,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.model_executor.models.utils import WeightsMapper

from vllm_xcpu_plugin.layers.fused_moe.grouped_gemm_experts import (
    XcpuGroupedGemmExperts,
)
from vllm_xcpu_plugin.layers.fused_moe.shared_experts import _packed_linear
from vllm_xcpu_plugin.layers.int8_ue8m0 import (
    XcpuInt8Ue8m0Config,
    XcpuInt8Ue8m0LinearMethod,
    XcpuInt8Ue8m0MoEMethod,
    register_xcpu_int8_quantization,
)


def test_xcpu_int8_config_registration_and_mapper() -> None:
    register_xcpu_int8_quantization()
    cfg_cls = get_quantization_config("xcpu_int8")
    assert cfg_cls is XcpuInt8Ue8m0Config

    cfg = cfg_cls.from_config({
        "quant_method": "xcpu_int8",
        "fmt": "int8_ue8m0",
        "activation_scheme": "dynamic",
        "weight_block_size": [16, 16],
        "modules_to_not_convert": ["lm_head", "model.embed_tokens"],
    })
    assert cfg.get_name() == "xcpu_int8"
    assert cfg.weight_block_size == [16, 16]
    assert cfg.expert_dtype is None
    assert cfg.get_supported_act_dtypes() == [torch.bfloat16]

    mapper = WeightsMapper(orig_to_new_prefix={"model.": "language_model."})
    cfg.apply_vllm_mapper(mapper)
    assert cfg.ignored_layers == ["lm_head", "language_model.embed_tokens"]

    cfg_fp4_experts = cfg_cls.from_config({
        "quant_method": "xcpu_int8",
        "fmt": "int8_ue8m0",
        "activation_scheme": "dynamic",
        "weight_block_size": [16, 16],
        "expert_dtype": "fp4",
    })
    assert cfg_fp4_experts.expert_dtype == "fp4"
    with pytest.raises(ValueError, match="expert_dtype"):
        XcpuInt8Ue8m0Config(expert_dtype="fp8")


def test_xcpu_int8_fp4_expert_dtype_returns_mxfp4_moe(monkeypatch) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    from vllm_xcpu_plugin.layers.fused_moe.routed_experts import xcpu_moe_method_factory
    from vllm_xcpu_plugin.layers.mxfp4_moe import XcpuMxfp4MoEMethod

    parallel_cfg = FusedMoEParallelConfig.make_no_parallel()
    moe_cfg = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=64,
        intermediate_size=64,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device="mcpu",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=parallel_cfg,
        in_dtype=torch.bfloat16,
        max_num_tokens=8,
    )

    class FakeRoutedExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe_config = moe_cfg

    monkeypatch.setattr(
        "vllm_xcpu_plugin.layers.int8_ue8m0.RoutedExperts", FakeRoutedExperts
    )
    cfg = XcpuInt8Ue8m0Config(weight_block_size=[16, 16], expert_dtype="fp4")
    layer = FakeRoutedExperts()
    fake_vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(quantization_config=None)
    )
    with set_current_vllm_config(fake_vllm_cfg):  # type: ignore[arg-type]
        quant_method = cfg.get_quant_method(layer, "layers.0.ffn.experts")
        assert type(quant_method) is Mxfp4MoEMethod
        xcpu_method = xcpu_moe_method_factory(quant_method, cfg, layer)
        assert type(xcpu_method) is XcpuMxfp4MoEMethod


def test_int8_grouped_wo_a_process_and_ops_dispatch(monkeypatch) -> None:
    import torch_xcpu

    from vllm_xcpu_plugin.models.deepseek_v41 import ops as dsv41_ops

    layer = torch.nn.Module()
    layer.xcpu_grouped_wo_a = 4
    layer.bias = None
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.ones((128, 64), dtype=torch.int8), requires_grad=False
        ),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.full((8, 4), 127, dtype=torch.uint8), requires_grad=False
        ),
    )

    prepack_calls = []

    def fake_prepack(w):
        prepack_calls.append(tuple(w.shape))
        return w.clone()

    monkeypatch.setattr(
        torch_xcpu.ops, "einsum_int8_tgd_grd_to_tgr_prepack_weight", fake_prepack
    )
    monkeypatch.setattr(
        torch_xcpu.ops, "einsum_int8_tgd_grd_to_tgr_version", lambda: "A0"
    )

    method = XcpuInt8Ue8m0LinearMethod(XcpuInt8Ue8m0Config(weight_block_size=[16, 16]))
    method.process_weights_after_loading(layer)

    assert prepack_calls == [(4, 32, 64)]
    assert hasattr(layer, "xcpu_wo_a")
    packed, scales = layer.xcpu_wo_a
    assert packed.shape == (4, 32, 64)
    assert scales.shape == (4, 2, 4)
    assert not hasattr(layer, "_xcpu_int8_linear")

    einsum_calls = []

    def fake_einsum_int8(x, w, s, out):
        einsum_calls.append((tuple(x.shape), w.dtype, s.dtype, tuple(out.shape)))
        out.fill_(1.0)

    monkeypatch.setattr(torch_xcpu.ops, "einsum_int8_tgd_grd_to_tgr", fake_einsum_int8)
    o = torch.zeros((3, 8, 32), dtype=torch.bfloat16)
    z = dsv41_ops.grouped_wo_a(o, packed, scales)
    assert z.shape == (3, 128)
    assert einsum_calls == [((3, 4, 64), torch.int8, torch.uint8, (3, 4, 32))]


def test_deepseek_v41_linear_scale_param_name_for_int8_32x32() -> None:
    import vllm.models.deepseek_v41.nvidia.dspark as nv_dspark
    import vllm.models.deepseek_v41.nvidia.model as nv_model
    import vllm.models.deepseek_v41.nvidia.vl_model as nv_vl_model

    from vllm_xcpu_plugin.models.deepseek_v41 import model as dsv41_model

    dsv41_model.install_deepseek_v41()

    int8_32_cfg = SimpleNamespace(
        quant_config=XcpuInt8Ue8m0Config(weight_block_size=[32, 32], expert_dtype="fp4")
    )
    for mod in (dsv41_model, nv_model, nv_vl_model, nv_dspark):
        assert mod._linear_scale_param_name(int8_32_cfg, "fp4") == "weight_scale_inv"

    mxfp8_cfg = SimpleNamespace(
        quant_config=SimpleNamespace(weight_block_size=[32, 32])
    )
    assert dsv41_model._linear_scale_param_name(mxfp8_cfg, "fp4") == "weight_scale"


def test_mla_weight_only_int8_ue8m0_dequant() -> None:
    layer = torch.nn.Module()
    layer._xcpu_mla_weight_only = True
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(
        torch.ones((3, 5), dtype=torch.int8), requires_grad=False
    )
    # E8M0 exponents: 127 -> 2^0 = 1, 128 -> 2^1 = 2, 129 -> 2^2 = 4, 130 -> 2^3 = 8
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.tensor([[127, 128], [129, 130]], dtype=torch.uint8),
        requires_grad=False,
    )
    cfg = XcpuInt8Ue8m0Config(weight_block_size=[2, 3])
    method = XcpuInt8Ue8m0LinearMethod(cfg)
    method.process_weights_after_loading(layer)

    expected = torch.tensor(
        [[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [4, 4, 4, 8, 8]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    assert not hasattr(layer, "_xcpu_int8_linear")


@pytest.mark.parametrize(
    "row_parallel,tp_rank,skip_bias_add,fuse_bias",
    [
        (False, 0, False, True),
        (False, 1, False, True),
        (False, 0, True, False),
        (True, 0, False, True),
        (True, 1, False, False),
        (True, 0, True, False),
    ],
)
def test_int8_linear_method_process_weights_and_shared_experts(
    monkeypatch,
    row_parallel: bool,
    tp_rank: int,
    skip_bias_add: bool,
    fuse_bias: bool,
) -> None:
    import torch_xcpu

    class Layer(RowParallelLinear if row_parallel else torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.tp_rank = tp_rank
            self.skip_bias_add = skip_bias_add
            self.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.zeros((64, 64), dtype=torch.int8), requires_grad=False
                ),
            )
            self.register_parameter(
                "weight_scale_inv",
                torch.nn.Parameter(
                    torch.full((4, 4), 127, dtype=torch.uint8), requires_grad=False
                ),
            )
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.ones((64,), dtype=torch.bfloat16), requires_grad=False
                ),
            )

    calls = []

    class FakeInt8Linear:
        def __init__(self, weight, weight_scale, scale_block_size, bias):
            self.params = SimpleNamespace(
                packed_weight=weight.clone(),
                weight_scale=weight_scale.clone(),
                scale_block_size=scale_block_size,
                bias=bias,
                backend=SimpleNamespace(name="INTEL_AMX", __int__=lambda s: 0),
            )

        def __call__(self, x):
            return x

    def fake_init(weight, weight_scale, scale_block_size, bias):
        calls.append((weight, weight_scale, scale_block_size, bias))
        return FakeInt8Linear(weight, weight_scale, scale_block_size, bias)

    monkeypatch.setattr(torch_xcpu.ops, "initialize_int8_linear", fake_init)
    layer = Layer()
    method = XcpuInt8Ue8m0LinearMethod(XcpuInt8Ue8m0Config())
    method.process_weights_after_loading(layer)

    assert len(calls) == 1
    assert calls[0][2] == (16, 16)
    if fuse_bias:
        assert calls[0][3] is not None and calls[0][3].dtype == torch.float32
        assert layer.bias.dtype == torch.float32
    else:
        assert calls[0][3] is None

    # Also verify shared_experts._packed_linear recognizes _xcpu_int8_linear
    # when bias is absent.
    layer.bias = None
    layer._xcpu_int8_linear.params.backend = 0
    packed = _packed_linear(layer)
    assert packed is not None
    assert packed[2] == 0
    assert packed[3] == (16, 16)


def test_grouped_gemm_experts_supports_int8_w8a16_scheme() -> None:
    int8_key = QuantKey(
        dtype=torch.int8,
        scale=ScaleDesc(dtype=torch.uint8, static=True, group_shape=GroupShape(16, 16)),
        symmetric=True,
    )
    assert XcpuGroupedGemmExperts._supports_quant_scheme(int8_key, None)


def test_xcpu_int8_moe_method_create_and_process(monkeypatch) -> None:
    import torch_xcpu

    parallel_cfg = FusedMoEParallelConfig.make_no_parallel()
    moe_cfg = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=64,
        intermediate_size=64,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device="mcpu",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=parallel_cfg,
        in_dtype=torch.bfloat16,
        max_num_tokens=8,
    )
    fake_layer = SimpleNamespace(moe_config=moe_cfg, apply_router_weight_on_input=False)
    method = XcpuInt8Ue8m0MoEMethod(XcpuInt8Ue8m0Config(), fake_layer)  # type: ignore[arg-type]

    class FakeRoutedExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe_config = moe_cfg

        def _expert_routing_tables(self):
            return None

    layer = FakeRoutedExperts()
    monkeypatch.setattr(
        "vllm_xcpu_plugin.layers.int8_ue8m0.RoutedExperts", FakeRoutedExperts
    )
    method.create_weights(
        layer,
        num_experts=4,
        hidden_size=64,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w13_weight.shape == (4, 128, 64)
    assert layer.w2_weight.shape == (4, 64, 64)
    assert layer.w13_weight_scale_inv.dtype == torch.uint8
    assert layer.w13_weight_scale_inv.shape == (4, 8, 4)
    assert layer.w2_weight_scale_inv.shape == (4, 4, 4)
    assert (
        layer.w13_weight_scale_inv.quant_method
        == FusedMoeWeightScaleSupported.BLOCK.value
    )

    init_calls = []
    install_calls = []

    def fake_init_fused_moe(
        w13, w2, w13_scale, w2_scale, *, scale_block_size, m_capacity, allocate_scratch
    ):
        init_calls.append((
            w13,
            w2,
            w13_scale,
            w2_scale,
            scale_block_size,
            m_capacity,
            allocate_scratch,
        ))
        return SimpleNamespace(
            params=SimpleNamespace(
                gemm1=SimpleNamespace(
                    params=SimpleNamespace(
                        packed_weight=w13, packed_weight_scale=w13_scale
                    )
                ),
                gemm2=SimpleNamespace(
                    params=SimpleNamespace(
                        packed_weight=w2, packed_weight_scale=w2_scale
                    )
                ),
            )
        )

    def fake_install_fused_moe(meth, lyr, fused_moe, make_qc, *, scale_names=None):
        qc = make_qc(lyr)
        install_calls.append((meth, lyr, qc, scale_names))

    monkeypatch.setattr(torch_xcpu.ops, "initialize_fused_moe", fake_init_fused_moe)
    monkeypatch.setattr(
        "vllm_xcpu_plugin.layers.int8_ue8m0.install_fused_moe", fake_install_fused_moe
    )

    method.process_weights_after_loading(layer)
    assert len(init_calls) == 1
    assert init_calls[0][4] == [16, 16]
    assert len(install_calls) == 1
    assert install_calls[0][3] == ("w13_weight_scale_inv", "w2_weight_scale_inv")
    assert install_calls[0][2].weight_quant_dtype == torch.int8


def test_sparse_indexer_patch_preserves_upstream_compatibility() -> None:
    from vllm_xcpu_plugin.attention.sparse_attn_indexer import (
        maybe_patch_vllm_sparse_attn_indexer,
    )
    from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility

    maybe_patch_vllm_sparse_attn_indexer()
    verify_upstream_compatibility(("sparse_indexer",))
