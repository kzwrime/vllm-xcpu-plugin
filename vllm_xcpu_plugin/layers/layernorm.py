import torch
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    LayerNorm,
    RMSNormGated,
)


def _not_implemented(message: str):
    raise NotImplementedError(f"torch_xcpu {message}")


@LayerNorm.register_oot
class XcpuLayerNorm(LayerNorm):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        import torch_xcpu.ops as ops

        if not x.is_contiguous() and not (x.dim() == 2 and x.stride(-1) == 1):
            _not_implemented(
                "LayerNorm only supports contiguous or row-strided 2D input"
            )
        return ops.layer_norm(x, self.weight.data, self.bias.data, self.eps)


@GemmaRMSNorm.register_oot
class XcpuGemmaRMSNorm(GemmaRMSNorm):
    @property
    def supports_packed_residual(self) -> bool:
        # A delayed aux sum must not bypass native dispatch or hooks that may
        # observe/change the inputs and outputs at the original norm boundary.
        from torch.nn.modules import module

        return (
            self._forward_method == self.forward_oot
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
