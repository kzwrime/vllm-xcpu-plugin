"""Shared post-load installation for all XCPU fused-MoE weight formats."""

from collections.abc import Callable

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.utils import replace_parameter

from .grouped_gemm_experts import XcpuGroupedGemmExperts


def reject_fused_moe_hot_reload(method) -> None:
    """Reject post-load processing after an XCPU MoE kernel is installed."""
    if method.moe_kernel is not None:
        raise RuntimeError("XCPU MoE hot weight updates are unsupported")


def install_fused_moe(
    method,
    layer,
    fused_moe,
    make_quant_config: Callable,
    *,
    scale_names: tuple[str, str] | None = None,
) -> None:
    """Publish packed tensors on the layer and install one Modular kernel."""
    gemm1 = fused_moe.params.gemm1.params
    gemm2 = fused_moe.params.gemm2.params
    replace_parameter(layer, "w13_weight", gemm1.packed_weight)
    replace_parameter(layer, "w2_weight", gemm2.packed_weight)
    if scale_names is not None:
        if gemm1.packed_weight_scale is None or gemm2.packed_weight_scale is None:
            raise RuntimeError("quantized XCPU MoE initialization returned no scales")
        replace_parameter(layer, scale_names[0], gemm1.packed_weight_scale)
        replace_parameter(layer, scale_names[1], gemm2.packed_weight_scale)

    layer._xcpu_fused_moe = fused_moe
    quant_config = make_quant_config(layer)
    if quant_config is None:
        raise RuntimeError("failed to construct XCPU fused-MoE quant config")
    method.moe_quant_config = quant_config

    prepare_finalize = maybe_make_prepare_finalize(
        moe=method.moe,
        quant_config=quant_config,
        routing_tables=layer._expert_routing_tables(),
        allow_new_interface=True,
        use_monolithic=False,
    )
    if not isinstance(prepare_finalize, mk.FusedMoEPrepareAndFinalizeModular):
        raise TypeError("XCPU fused MoE requires Modular Prepare/Finalize")
    method.moe_kernel = mk.FusedMoEKernel(
        prepare_finalize,
        XcpuGroupedGemmExperts(method.moe, quant_config, fused_moe),
    )
