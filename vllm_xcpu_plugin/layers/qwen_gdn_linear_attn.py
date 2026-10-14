import torch
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from vllm_xcpu_plugin.gdn_metadata_patch import XcpuGDNAttentionMetadata


@QwenGatedDeltaNetAttention.register_oot
class XcpuQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xcpu_native_core = envs.VLLM_XCPU_GDN_COMPILE
        self._xcpu_conv_state_dim_first = is_conv_state_dim_first()
        if self._xcpu_native_core and self._xcpu_conv_state_dim_first:
            raise ValueError(
                "Native XCPU GDN requires VLLM_SSM_CONV_STATE_LAYOUT=SD; "
                "the XCPU convolution kernels do not support DS layout."
            )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        if self._xcpu_native_core:
            # All v0.31 cache groups share one backing allocation. Use the same
            # carrier dtype as full attention to avoid mixed-dtype mutation
            # aliases in AOTAutograd. This is a byte-preserving view, not a cast.
            assert self.kv_cache[1].dtype == torch.float32
            assert self.kv_cache[0].dtype == self.model_config.dtype
            carrier_dtype = kv_cache_dtype_str_to_dtype(
                self.cache_config.cache_dtype, self.model_config
            )
            self._xcpu_packed_kv_cache = kv_cache.view(carrier_dtype)
            self._xcpu_conv_shape = tuple(self.kv_cache[0].shape[1:])

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._xcpu_native_core:
            return super().forward_cuda(hidden_states)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        if self.gqa_interleaved_layout:
            q, k, v, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
            mixed_qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], -1)
        else:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            mixed_qkv, z = mixed_qkvz.split(
                [qkv_size, self.value_dim // self.tp_size], dim=-1
            )
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)
        core_attn_out = torch.empty(
            (hidden_states.size(0), self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward_core_native(mixed_qkv, b, a, core_attn_out)
        return self._output_projection(core_attn_out, z)

    def _forward_core_native(self, mixed_qkv, b, a, core_attn_out):
        metadata = get_forward_context().attn_metadata
        m = metadata.get(self.prefix) if isinstance(metadata, dict) else None
        if m is None:
            core_attn_out.zero_()
            return
        assert isinstance(m, XcpuGDNAttentionMetadata)
        assert m.xcpu_core_counts is not None
        torch.ops.torch_xcpu.qwen_gdn_attention_core_state(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self._xcpu_packed_kv_cache,
            self.A_log,
            self.dt_bias,
            self.conv1d.weight,
            self.conv1d.bias,
            m.xcpu_core_counts,
            *m.xcpu_core_tensors,
            self.num_k_heads // self.tp_size,
            self.head_k_dim,
            self.head_v_dim,
            *self._xcpu_conv_shape,
            self._xcpu_conv_state_dim_first,
            self.enable_packed_recurrent_decode,
            self.activation in ("silu", "swish"),
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
        del chunk_indices, chunk_offsets

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
            if core_attn_out is not None:
                core_attn_out.view(-1)[: output.numel()].copy_(output.reshape(-1))
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

        output, final_state = torch_xcpu.ops.chunk_gated_delta_rule_separated(
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
        if core_attn_out is not None:
            core_attn_out.view(-1)[: output.numel()].copy_(output.reshape(-1))
        return output, final_state

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
