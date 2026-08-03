# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    convert_to_unquantized_kernel_format,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.utils import replace_parameter

from .cpu_groupgemm_moe_v2 import CPUGroupGemmExperts

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


@UnquantizedFusedMoEMethod.register_oot
class XcpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

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
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=routing_tables,
            allow_new_interface=True,
            use_monolithic=False,
        )
        assert isinstance(prepare_finalize, mk.FusedMoEPrepareAndFinalizeModular)
        return prepare_finalize

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
