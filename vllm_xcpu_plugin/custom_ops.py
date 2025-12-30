from typing import Optional, Union
import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm, fused_add_rms_norm, rms_norm
from vllm.logger import logger

@RMSNorm.register_oot
class XcpuRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        logger.info("Init XcpuRMSNorm")

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # logger.debug("XcpuRMSNorm.forward_cpu")

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            return fused_add_rms_norm(x, residual, self.weight.data,
                                      self.variance_epsilon)
        else:
            return rms_norm(x, self.weight.data, self.variance_epsilon)

