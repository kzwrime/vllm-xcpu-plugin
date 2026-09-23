# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention import mla_attention


class _Impl:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported
        self.call_args = None

    def fused_mla_rope_kvcache_supported(self) -> bool:
        return self.supported

    def do_fused_mla_rope_kvcache_update(self, *args) -> None:
        self.call_args = args


class _Layer:
    fused_mla_rope_kvcache_supported = (
        mla_attention.MLAAttention.fused_mla_rope_kvcache_supported
    )
    maybe_fused_mla_rope_kvcache_update = (
        mla_attention.MLAAttention.maybe_fused_mla_rope_kvcache_update
    )

    def __init__(self, impl: _Impl) -> None:
        self.impl = impl
        self.use_direct_call = True
        self.use_pcp = False
        self.calculate_kv_scales = False
        self.layer_name = "layers.0.attn"
        self.kv_cache = torch.empty(1)
        self.kv_cache_dtype = "auto"
        self._k_scale = torch.ones(1)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("use_direct_call", False),
        ("use_pcp", True),
        ("calculate_kv_scales", True),
    ],
)
def test_fused_mla_rope_cache_ignores_non_target_paths(attribute, value) -> None:
    layer = _Layer(_Impl())
    setattr(layer, attribute, value)
    assert not layer.fused_mla_rope_kvcache_supported()


def test_fused_mla_rope_cache_requires_backend_opt_in() -> None:
    layer = _Layer(_Impl(supported=False))
    assert not layer.fused_mla_rope_kvcache_supported()


def test_fused_mla_rope_cache_leaves_shape_validation_to_op(monkeypatch) -> None:
    impl = _Impl()
    layer = _Layer(impl)
    slots = torch.tensor([0, 1], dtype=torch.int64)
    monkeypatch.setattr(
        mla_attention,
        "get_forward_context",
        lambda: SimpleNamespace(slot_mapping={layer.layer_name: slots}),
    )

    # Deliberately use dimensions outside the current D576/V512 kernel. The
    # routing layer must still invoke the backend so its operator can report
    # the unsupported shape instead of silently selecting the unfused path.
    positions = torch.arange(2)
    q_pe = torch.empty(2, 3, 16)
    k_pe = torch.empty(2, 1, 16)
    kv_c = torch.empty(2, 32)
    cos_sin = torch.empty(64, 16)

    assert layer.maybe_fused_mla_rope_kvcache_update(
        positions, q_pe, k_pe, kv_c, cos_sin, False
    )
    assert impl.call_args is not None
    assert impl.call_args[0] is q_pe
    assert impl.call_args[1] is k_pe
    assert impl.call_args[2] is kv_c
    assert torch.equal(impl.call_args[7], slots)


def test_qproj_plugin_normalizes_explicit_bfloat16_cache(monkeypatch):
    import torch_xcpu

    from vllm_xcpu_plugin.flashattn_mla_sparse_patch import (
        _xcpu_do_fused_mla_rope_qproj_kvcache_update,
    )

    calls = []
    monkeypatch.setattr(
        torch_xcpu.ops,
        "fused_mla_rope_qproj_cat_cache",
        lambda *args: calls.append(args),
    )
    q = torch.empty(2, 8, 128)
    k = torch.empty(2, 1, 64)
    _xcpu_do_fused_mla_rope_qproj_kvcache_update(
        None,
        q,
        torch.empty(8, 64, 512),
        k,
        torch.empty(2, 512),
        torch.arange(2),
        torch.empty(4, 64),
        False,
        torch.empty(1, 16, 576),
        torch.arange(2),
        "bfloat16",
        torch.ones(1),
        torch.empty(2, 8, 576),
    )
    assert calls[0][-2] == "auto"
    assert calls[0][2].shape == (2, 64)
