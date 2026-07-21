import torch
from vllm import envs
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
)


@ChunkGatedDeltaRule.register_oot
class XcpuChunkGatedDeltaRule(ChunkGatedDeltaRule):
    def __init__(self) -> None:
        super().__init__()
        self.enable_custom_prefill = envs.VLLM_ENABLE_FLA_CUSTOM_PREFILL
        if self.enable_custom_prefill:
            self._forward_method = self.forward_custom_v2
        else:
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

        assert cu_seqlens is not None
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

    def forward_custom(
        self,
        mixed_qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        import torch_xcpu

        assert cu_seqlens is not None
        num_seqs = cu_seqlens.shape[0] - 1
        _, HV, V, K = ssm_state.shape
        T = mixed_qkv.shape[0]
        C = 64
        max_chunk_num = (T + C - 1) / C + num_seqs
        max_workspace_size = max_chunk_num * (
            HV * (C * V * 2 + C * C + K * V + C * K) + 3 * 2
        )
        workspace = torch.empty(
            int(max_workspace_size), dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )

        return torch_xcpu.ops.chunk_gated_delta_rule_separated_custom(
            mixed_qkv=mixed_qkv,
            g=g,
            beta=beta,
            ssm_state=ssm_state,
            ssm_state_indices=ssm_state_indices,
            has_initial_state=has_initial_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            workspace=workspace,
        )
    
    def forward_custom_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        import torch_xcpu

        assert cu_seqlens is not None
        num_seqs = cu_seqlens.shape[0] - 1
        _, HV, V, K = ssm_state.shape
        T = q.shape[1]
        C = 64
        max_chunk_num = (T + C - 1) / C + num_seqs
        max_workspace_size = max_chunk_num * (
            HV * (C * V * 2 + C * C + K * V + C * K) + 3 * 2
        )
        workspace = torch.empty(
            int(max_workspace_size), dtype=q.dtype, device=q.device
        )

        return torch_xcpu.ops.chunk_gated_delta_rule_separated_custom_v2(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_state=ssm_state,
            ssm_state_indices=ssm_state_indices,
            has_initial_state=has_initial_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            workspace=workspace,
        )