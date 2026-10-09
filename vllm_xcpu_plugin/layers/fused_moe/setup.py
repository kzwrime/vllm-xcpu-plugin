"""Shared post-load installation for all XCPU fused-MoE weight formats."""

from collections.abc import Callable

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.utils import replace_parameter

from vllm_xcpu_plugin import envs

from .grouped_gemm_experts import XcpuGroupedGemmExperts

logger = init_logger(__name__)


def reject_fused_moe_hot_reload(method) -> None:
    """Reject post-load processing after an XCPU MoE kernel is installed."""
    if method.moe_kernel is not None:
        raise RuntimeError("XCPU MoE hot weight updates are unsupported")


def use_fused_ep_moe(moe) -> bool:
    parallel = moe.moe_parallel_config
    return (
        envs.VLLM_XCPU_ENABLE_FUSED_EP_MOE
        and parallel.use_ep
        and parallel.all2all_backend in {"mpi_alltoallv_v5", "mpi_alltoallv_v6"}
    )


def use_shared_moe_workspace(moe) -> bool:
    return use_fused_ep_moe(moe) or moe.moe_parallel_config.all2all_backend in {
        "mpi_alltoallv_v7",
        "mpi_alltoallv_v8",
    }


def install_fused_moe(
    method,
    layer,
    fused_moe,
    make_quant_config: Callable,
    *,
    scale_names: tuple[str, str] | None = None,
) -> None:
    """Publish packed tensors and install local compute or complete EP execution."""
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
    experts = XcpuGroupedGemmExperts(method.moe, quant_config, fused_moe)
    parallel = method.moe.moe_parallel_config
    if use_fused_ep_moe(method.moe):
        from .ep_experts import XcpuEPExperts, XcpuEPKernelAdapter

        if layer.expert_map is None:
            raise ValueError("XCPU EP requires an expert map")
        layer._xcpu_ep_experts = XcpuEPExperts(
            fused_moe,
            prepare_finalize._comm_metadata,
            prepare_finalize.comm_ptr_wrapper,
            layer.expert_map,
            version=int(prepare_finalize.version[1:]),
            ep_size=prepare_finalize.ep_size,
            ep_rank=prepare_finalize.ep_rank,
            max_num_tokens=prepare_finalize.max_moe_tokens_per_rank,
            topk=method.moe.experts_per_token,
        )
        method.moe_kernel = XcpuEPKernelAdapter(
            prepare_finalize, experts, layer._xcpu_ep_experts
        )
        logger.info_once(
            "Using XcpuEPExperts: version=%s transport=%s compute=%s",
            layer._xcpu_ep_experts.version,
            parallel.all2all_backend,
            fused_moe.resolved_backend,
            scope="process",
        )
    else:
        method.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
