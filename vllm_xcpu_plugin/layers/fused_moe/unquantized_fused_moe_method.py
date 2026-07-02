# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    convert_to_unquantized_kernel_format,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.utils import replace_parameter

import vllm_xcpu_plugin.envs as envs_xcpu

from .cpu_groupgemm_moe_v2 import CPUGroupGemmExperts
from .mpi_alltoallv_prepare_finalize_v1 import MpiAlltoallvPrepareAndFinalizeV1
from .mpi_alltoallv_prepare_finalize_v2 import MpiAlltoallvPrepareAndFinalizeV2
from .mpi_alltoallv_prepare_finalize_v3 import MpiAlltoallvPrepareAndFinalizeV3
from .mpi_alltoallv_prepare_finalize_v4 import MpiAlltoallvPrepareAndFinalizeV4
from .torch_all_to_all_single_prepare_finalize import (
    TorchAlltoallSinglePrepareAndFinalize,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

# Map version strings to implementation classes
_MPI_ALLTOALLV_VERSIONS: dict[str, type[Any]] = {
    "v1": MpiAlltoallvPrepareAndFinalizeV1,
    "v2": MpiAlltoallvPrepareAndFinalizeV2,
    "v3": MpiAlltoallvPrepareAndFinalizeV3,
    "v4": MpiAlltoallvPrepareAndFinalizeV4,
}


@UnquantizedFusedMoEMethod.register_oot
class XcpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def _experts_reduce_topk(self) -> bool:
        # No-DP/EP keeps the original [M, topk] routing layout, so the CPU
        # experts must apply router weights and reduce to [M, hidden_size].
        # XCPU all2all prepare/finalize implementations expand routed tokens
        # before expert compute and reduce them after reverse communication.
        return not self.moe.moe_parallel_config.use_all2all_kernels

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def supports_eplb(self) -> bool:
        return True

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular:
        # EPLB is applied by the router before the MoE kernel sees topk_ids, so
        # XCPU all2all can keep using linear physical expert ids. Non-empty
        # routing tables indicate round-robin placement, which XCPU all2all does
        # not support yet.
        if routing_tables is not None:
            raise NotImplementedError(
                "XCPU MoE supports EPLB with linear physical expert placement, "
                "but does not support round-robin routing tables."
            )

        if not self.moe.moe_parallel_config.use_all2all_kernels:
            return MoEPrepareAndFinalizeNoDPEPModular()

        _ep_group = get_ep_group()
        assert _ep_group is not None
        assert _ep_group.device_communicator is not None
        all2all_manager = _ep_group.device_communicator.all2all_manager
        assert all2all_manager is not None

        ep_group = (
            _ep_group.device_communicator.device_group
            if _ep_group.device_communicator.device_group is not None
            else _ep_group.device_communicator.cpu_group
        )
        assert ep_group is not None
        num_dispatchers = all2all_manager.world_size
        rank_expert_offset = all2all_manager.rank * self.moe.num_local_experts

        if envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU == "torch_all_to_all_single":
            return TorchAlltoallSinglePrepareAndFinalize(
                ep_group=ep_group,
                num_local_experts=self.moe.num_local_experts,
                num_dispatchers=num_dispatchers,
                rank_expert_offset=rank_expert_offset,
                tp_rank=self.moe.tp_rank,
                tp_size=self.moe.tp_size,
            )

        if envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU == "mpi_alltoallv":
            version = envs_xcpu.VLLM_MPI_ALLTOALLV_VERSION
            mpi_impl_class = _MPI_ALLTOALLV_VERSIONS.get(version)
            if mpi_impl_class is None:
                raise ValueError(
                    f"Invalid VLLM_MPI_ALLTOALLV_VERSION: {version}. "
                    f"Must be one of: {list(_MPI_ALLTOALLV_VERSIONS.keys())}"
                )

            mpi_prepare_finalize_kwargs: dict[str, Any] = {
                "max_num_tokens": self.moe.max_num_tokens,
                "ep_group": ep_group,
                "num_experts": self.moe.num_experts,
                "num_local_experts": self.moe.num_local_experts,
                "num_dispatchers": num_dispatchers,
                "rank_expert_offset": rank_expert_offset,
                "dp_rank": self.moe.dp_rank,
                "dp_size": self.moe.dp_size,
            }
            if version == "v4":
                mpi_prepare_finalize_kwargs["is_sequence_parallel"] = (
                    self.moe.moe_parallel_config.is_sequence_parallel
                )
                assert self.moe.moe_parallel_config.is_sequence_parallel
            return mpi_impl_class(**mpi_prepare_finalize_kwargs)

        raise ValueError(
            "Unsupported VLLM_ALL2ALL_BACKEND_XCPU for MoE: "
            f"{envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU}"
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: "RoutedExperts",
    ) -> mk.FusedMoEExpertsModular:
        if self.moe_quant_config is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        fused_experts = CPUGroupGemmExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            topk_reduce=self._experts_reduce_topk(),
        )
        assert (
            prepare_finalize.activation_format == fused_experts.activation_format()
        ), (
            f"prepare_finalize.activation_format {prepare_finalize.activation_format}"
            f" != fused_experts.activation_format() {fused_experts.activation_format()}"
        )
        return fused_experts

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format if needed.
        w13_new, w2_new = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            moe_config=layer.moe_config,
            w13_weight=w13,
            w2_weight=w2,
        )
        # `moe_kernel` is initialized to None in FusedMoEMethodBase.__init__;
        # On the first call we replace the parameter normally. On subsequent
        # calls (e.g. RL weight updates that re-trigger
        # process_weights_after_loading) the moe kernel has already been set
        # up and CUDA graphs may have captured the parameter addresses, so
        # we copy the shuffled data into the existing storage instead of
        # re-registering a new Parameter.
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]

        assert not is_weight_update, "XCPU MoE hot weight update unsupported"

        replace_parameter(layer, "w13_weight", w13_new, prefer_copy=is_weight_update)
        replace_parameter(layer, "w2_weight", w2_new, prefer_copy=is_weight_update)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        prepare_finalize = self.maybe_make_prepare_finalize(
            routing_tables=layer._expert_routing_tables(),
        )

        self.moe_kernel = mk.FusedMoEKernel(
            prepare_finalize,
            self.select_gemm_impl(prepare_finalize, layer),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        assert isinstance(layer, RoutedExperts)
        super().process_weights_after_loading(layer)

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        assert isinstance(w13, torch.Tensor)
        assert isinstance(w2, torch.Tensor)

        self._setup_kernel(
            layer=layer,
            w13=w13,
            w2=w2,
        )
