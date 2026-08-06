import pytest
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_xcpu_plugin.layers.fp8_moe import XcpuFp8MoEMethod
from vllm_xcpu_plugin.layers.fused_moe.unquantized_fused_moe_method import (
    XcpuUnquantizedFusedMoEMethod,
)
from vllm_xcpu_plugin.layers.mxfp4_moe import XcpuCompressedTensorsMxfp4MoEMethod


@pytest.mark.parametrize(
    "method_cls",
    (
        XcpuUnquantizedFusedMoEMethod,
        XcpuFp8MoEMethod,
        XcpuCompressedTensorsMxfp4MoEMethod,
    ),
)
def test_xcpu_moe_methods_reject_hot_reload_before_processing(method_cls):
    method = object.__new__(method_cls)
    method.moe_kernel = object()
    layer = object.__new__(RoutedExperts)

    with pytest.raises(RuntimeError, match="hot weight updates are unsupported"):
        method.process_weights_after_loading(layer)
