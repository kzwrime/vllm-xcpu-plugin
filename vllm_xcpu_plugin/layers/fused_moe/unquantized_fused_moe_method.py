# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from torch.nn import Module
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

from .cpu_groupgemm_moe_v2 import CPUGroupGemmExperts


@UnquantizedFusedMoEMethod.register_oot
class XcpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.topk_reduce = True

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
