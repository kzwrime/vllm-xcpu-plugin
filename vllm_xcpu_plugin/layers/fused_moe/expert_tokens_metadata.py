from dataclasses import dataclass

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk


@dataclass
class XCPUExpertTokensMetadata(mk.ExpertTokensMetadata):
    """Routing metadata shared by EP with 2D token XCPU MoE backends."""

    num_input_rows_valid: torch.Tensor | None = None
