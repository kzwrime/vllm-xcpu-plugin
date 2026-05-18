# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from torch.nn import Module
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
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
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

import vllm_xcpu_plugin.envs as envs_xcpu

from .cpu_groupgemm_moe_v2 import CPUGroupGemmExperts
from .mpi_alltoallv_prepare_finalize_v1 import MpiAlltoallvPrepareAndFinalizeV1
from .mpi_alltoallv_prepare_finalize_v2 import MpiAlltoallvPrepareAndFinalizeV2
from .mpi_alltoallv_prepare_finalize_v3 import MpiAlltoallvPrepareAndFinalizeV3
from .torch_all_to_all_single_prepare_finalize import (
    TorchAlltoallSinglePrepareAndFinalize,
)

# Map version strings to implementation classes
_MPI_ALLTOALLV_VERSIONS = {
    "v1": MpiAlltoallvPrepareAndFinalizeV1,
    "v2": MpiAlltoallvPrepareAndFinalizeV2,
    "v3": MpiAlltoallvPrepareAndFinalizeV3,
}


@UnquantizedFusedMoEMethod.register_oot
class XcpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.topk_reduce = envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU != "mpi_alltoallv"

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        prepare_finalize = super().maybe_make_prepare_finalize(routing_tables)

        _ep_group = get_ep_group()
        assert _ep_group is not None
        assert _ep_group.device_communicator is not None
        all2all_manager = _ep_group.device_communicator.all2all_manager

        if self.moe.moe_parallel_config.use_all2all_kernels:
            assert all2all_manager is not None
            if envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU == "torch_all_to_all_single":
                ep_group = (
                    _ep_group.device_communicator.device_group
                    if _ep_group.device_communicator.device_group is not None
                    else _ep_group.device_communicator.cpu_group
                )
                assert ep_group is not None
                num_dispatchers = all2all_manager.world_size

                prepare_finalize = TorchAlltoallSinglePrepareAndFinalize(
                    ep_group=ep_group,
                    num_local_experts=self.moe.num_local_experts,
                    num_dispatchers=num_dispatchers,
                    rank_expert_offset=all2all_manager.rank
                    * self.moe.num_local_experts,
                    tp_rank=self.moe.tp_rank,
                    tp_size=self.moe.tp_size,
                )
            elif envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU == "mpi_alltoallv":
                ep_group = (
                    _ep_group.device_communicator.device_group
                    if _ep_group.device_communicator.device_group is not None
                    else _ep_group.device_communicator.cpu_group
                )
                assert ep_group is not None
                num_dispatchers = all2all_manager.world_size

                # Select MPI alltoallv implementation based on environment variable
                version = envs_xcpu.VLLM_MPI_ALLTOALLV_VERSION
                mpi_impl_class = _MPI_ALLTOALLV_VERSIONS.get(version)
                if mpi_impl_class is None:
                    raise ValueError(
                        f"Invalid VLLM_MPI_ALLTOALLV_VERSION: {version}. "
                        f"Must be one of: {list(_MPI_ALLTOALLV_VERSIONS.keys())}"
                    )

                prepare_finalize = mpi_impl_class(
                    max_num_tokens=self.moe.max_num_tokens,
                    ep_group=ep_group,
                    num_experts=self.moe.num_experts,
                    num_local_experts=self.moe.num_local_experts,
                    num_dispatchers=num_dispatchers,
                    rank_expert_offset=all2all_manager.rank
                    * self.moe.num_local_experts,
                    dp_rank=self.moe.dp_rank,
                    dp_size=self.moe.dp_size,
                )
            else:
                pass

        return prepare_finalize

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            raise NotImplementedError("BatchedExperts not supported")
        else:
            # logger.debug("CPUGroupGemmExperts %s", self.moe)
            return CPUGroupGemmExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                topk_reduce=self.topk_reduce,
            )

    def _setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2 = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            layer=layer,
            w13_weight=w13,
            w2_weight=w2,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        self.kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            CPUGroupGemmExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                topk_reduce=self.topk_reduce,
            ),
            inplace=False,
        )

    def _maybe_prepack_grouped_gemm(self, layer: torch.nn.Module) -> None:
        if current_platform.get_cpu_architecture() != CpuArchEnum.X86:
            return

        from torch_xcpu import ops as xcpu_ops

        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight
        assert isinstance(w13_weight, torch.Tensor)
        assert isinstance(w2_weight, torch.Tensor)

        # fused_moe_compute passes trans_b=True for both grouped GEMMs, so the
        # weights are interpreted as [num_experts, output_size, input_size].
        xcpu_ops.prepack_moe_grouped_gemm(
            w13_weight,
            trans_b=True,
            output_size=w13_weight.size(1),
        )
        xcpu_ops.prepack_moe_grouped_gemm(
            w2_weight,
            trans_b=True,
            output_size=w2_weight.size(1),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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
        self._maybe_prepack_grouped_gemm(layer)
