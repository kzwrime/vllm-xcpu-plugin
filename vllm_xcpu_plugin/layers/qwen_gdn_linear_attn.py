import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
)


@ChunkGatedDeltaRule.register_oot
class XcpuChunkGatedDeltaRule(ChunkGatedDeltaRule):
    def __init__(self) -> None:
        super().__init__()
        self._forward_method = self.forward_oot

    def forward_oot(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        import torch_xcpu

        assert cu_seqlens is not None
        output, final_state = torch_xcpu.ops.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            output_flat = output.squeeze(0).reshape(-1)
            core_attn_out_flat = core_attn_out.reshape(-1)
            core_attn_out_flat[: output_flat.numel()].copy_(output_flat)
        return output, final_state
