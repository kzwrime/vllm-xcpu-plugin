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
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        ssm_state: torch.Tensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
    ):
        import torch_xcpu

        assert cu_seqlens is not None
        del chunk_indices, chunk_offsets, core_attn_out

        if ssm_state is not None:
            assert initial_state is None
            assert output_final_state
            assert ssm_state_indices is not None
            assert has_initial_state is not None
            _, HV, V, K = ssm_state.shape
            T = q.shape[1]
            workspace = self._allocate_workspace(q, cu_seqlens, HV, V, K, T)
            output = torch_xcpu.ops.chunk_gated_delta_rule_separated_custom_v2(
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
            return output, None

        assert initial_state is not None
        assert ssm_state_indices is None
        assert has_initial_state is None
        B, T, H, K = q.shape
        HV = v.shape[2]
        V = v.shape[3]
        assert B == 1
        del H
        workspace = self._allocate_workspace(q, cu_seqlens, HV, V, K, T)

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

    @staticmethod
    def _allocate_workspace(
        q: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_v_heads: int,
        value_dim: int,
        key_dim: int,
        num_tokens: int,
    ) -> torch.Tensor:
        num_seqs = cu_seqlens.shape[0] - 1
        C = 64
        max_chunk_num = (num_tokens + C - 1) / C + num_seqs
        max_workspace_size = max_chunk_num * (
            num_v_heads
            * (C * value_dim * 2 + C * C + key_dim * value_dim + C * key_dim)
            + 3 * 2
        )
        return torch.empty(
            int(max_workspace_size),
            dtype=q.dtype,
            device=q.device,
        )
