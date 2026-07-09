import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated


def _not_implemented(message: str):
    raise NotImplementedError(f"torch_xcpu {message}")


@GemmaRMSNorm.register_oot
class XcpuGemmaRMSNorm(GemmaRMSNorm):
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
