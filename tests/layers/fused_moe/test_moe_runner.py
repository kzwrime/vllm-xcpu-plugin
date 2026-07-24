from types import SimpleNamespace

from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_shared,
)

from vllm_xcpu_plugin.layers.fused_moe.moe_runner import XcpuMoERunner


def test_xcpu_moe_runner_is_registered_out_of_tree():
    assert op_registry_oot[MoERunner.__name__] is XcpuMoERunner


def test_xcpu_moe_runner_selects_direct_forward():
    without_shared = SimpleNamespace(_shared_experts=None)
    with_shared = SimpleNamespace(_shared_experts=object())

    assert XcpuMoERunner._select_forward(without_shared) is _moe_forward
    assert XcpuMoERunner._select_forward(with_shared) is _moe_forward_shared
