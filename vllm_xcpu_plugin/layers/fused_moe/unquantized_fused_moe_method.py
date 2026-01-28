# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.platforms import current_platform

import vllm_xcpu_plugin.envs as envs_xcpu

from .cpu_groupgemm_moe_v1 import CPUGroupGemmExperts
from .torch_all_to_all_single_prepare_finalize import (
    TorchAlltoallSinglePrepareAndFinalize,
)


# --8<-- [start:unquantized_fused_moe]
@UnquantizedFusedMoEMethod.register_oot
class XcpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    # --8<-- [end:unquantized_fused_moe]

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        # logger.info("Using XcpuUnquantizedFusedMoEMethod")

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalize | None:
        prepare_finalize = super().maybe_make_prepare_finalize(routing_tables)

        all2all_manager = get_ep_group().device_communicator.all2all_manager

        assert all2all_manager is not None

        if self.moe.moe_parallel_config.use_all2all_kernels:
            if envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU == "torch_all_to_all_single":
                _ep_group = get_ep_group()
                assert _ep_group is not None
                ep_group = (
                    _ep_group.device_communicator.device_group
                    if _ep_group.device_communicator.device_group is not None
                    else _ep_group.device_communicator.cpu_group
                )
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
            else:
                pass

        return prepare_finalize

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            raise NotImplementedError("BatchedExperts not supported")
        else:
            logger.debug("CPUGroupGemmExperts %s", self.moe)
            return CPUGroupGemmExperts(layer, self.moe_quant_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        self.use_inplace = True
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        self.kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            CPUGroupGemmExperts(layer, self.moe_quant_config),
            shared_experts=None,
        )

    def forward_cpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            layer.enable_eplb is not False
            or layer.expert_load_view is not None
            or layer.logical_to_physical_map is not None
            or layer.logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for CPU.")

        topk_weights, topk_ids = router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        result = self.kernel(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )

        return result

    assert current_platform.is_cpu()
    forward_native = forward_cpu
