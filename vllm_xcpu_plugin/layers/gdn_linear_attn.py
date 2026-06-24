import torch
from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule


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
        use_qk_l2norm_in_kernel: bool = True,
    ):
        import torch_xcpu

        num_seqs = cu_seqlens.shape[0] - 1
        B, T, H, K = q.shape
        HV = v.shape[2]
        V = v.shape[3]
        assert B == 1
        C = 64
        max_chunk_num = (T + C - 1) / C + num_seqs
        max_workspace_size = max_chunk_num * (
            HV * (C * V * 2 + C * C + K * V + C * K) + 3 * 2
        )
        workspace = torch.empty(int(max_workspace_size), dtype=q.dtype, device=q.device)

        return torch_xcpu.ops.chunk_gated_delta_rule_separated(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            workspace=workspace,
        )
