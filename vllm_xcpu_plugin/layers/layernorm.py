from typing import NoReturn

import torch
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    LayerNorm,
    RMSNormGated,
)


def _not_implemented(message: str) -> NoReturn:
    raise NotImplementedError(f"torch_xcpu {message}")


@LayerNorm.register_oot
class XcpuLayerNorm(LayerNorm):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if weight is None or bias is None:
            _not_implemented(
                "LayerNorm without affine weight or bias requires torch_xcpu "
                "layer_norm support for optional parameters"
            )
        if weight.dtype != torch.float32 or bias.dtype != torch.float32:
            _not_implemented(
                "LayerNorm parameter dtypes other than float32 require "
                "torch_xcpu layer_norm dtype support"
            )
        if x.dtype not in (torch.bfloat16, torch.float32):
            _not_implemented(
                f"LayerNorm input dtype {x.dtype} requires torch_xcpu "
                "layer_norm dtype support"
            )
        if not x.is_contiguous() and not (x.dim() == 2 and x.stride(-1) == 1):
            _not_implemented(
                "LayerNorm only supports contiguous or row-strided 2D input; "
                "torch_xcpu layer_norm needs broader layout support"
            )

        import torch_xcpu.ops as ops

        return ops.layer_norm(x, weight.data, bias.data, self.eps)


@GemmaRMSNorm.register_oot
class XcpuGemmaRMSNorm(GemmaRMSNorm):
    @property
    def supports_packed_residual(self) -> bool:
        # A delayed aux sum must not bypass native dispatch or hooks that may
        # observe/change the inputs and outputs at the original norm boundary.
        from torch.nn.modules import module

        # Dynamo does not preserve equality of separately accessed bound methods.
        # Compare their receiver and function identities instead.
        return (
            getattr(self._forward_method, "__self__", None) is self
            and getattr(self._forward_method, "__func__", None)
            is self.forward_oot.__func__
            and not self._forward_pre_hooks
            and not self._forward_hooks
            and not module._global_forward_pre_hooks
            and not module._global_forward_hooks
        )

    def forward_with_residual_out(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        residual_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_xcpu.ops as ops

        return ops.gemma_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual,
            residual_out=residual_out,
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_xcpu.ops as ops

        if x.stride(-1) != 1 or (residual is not None and residual.stride(-1) != 1):
            _not_implemented(
                "GemmaRMSNorm only supports input/residual "
                "with contiguous last dimension"
            )
        return ops.gemma_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual,
        )


@RMSNormGated.register_oot
class XcpuRMSNormGated(RMSNormGated):
    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_xcpu.ops as ops

        if self.activation not in ["silu", "swish"]:
            _not_implemented(
                f"RMSNormGated only supports activation='swish', "
                f"got {self.activation!r}"
            )
        if x.stride(-1) != 1 or (z is not None and z.stride(-1) != 1):
            _not_implemented(
                "RMSNormGated only supports input/gate with contiguous last dimension"
            )
        return ops.rms_norm_gated(
            x,
            self.weight.data,
            z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            epsilon=self.eps,
        )
