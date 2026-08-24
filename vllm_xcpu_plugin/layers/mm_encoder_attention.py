"""XCPU multimodal encoder attention integration."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)


@MMEncoderAttention.register_oot
class XcpuMMEncoderAttention(MMEncoderAttention):
    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del max_seqlen, sequence_lengths
        batch_size, query_length = query.shape[:2]
        key_value_length = key.shape[1]
        was_flattened = query.dim() != 4
        query, key, value = self.view_qkv_to_4d(
            query,
            key,
            value,
            batch_size,
            query_length,
            key_value_length,
        )

        import torch_xcpu

        if cu_seqlens is None:
            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * query_length,
                step=query_length,
                dtype=torch.int32,
                device=query.device,
            )
            if query_length == key_value_length:
                cu_seqlens_k = cu_seqlens_q
            else:
                cu_seqlens_k = torch.arange(
                    0,
                    (batch_size + 1) * key_value_length,
                    step=key_value_length,
                    dtype=torch.int32,
                    device=key.device,
                )
        else:
            cu_seqlens_q = cu_seqlens
            cu_seqlens_k = cu_seqlens

        packed_query = query.reshape(-1, query.shape[-2], query.shape[-1])
        packed_key = key.reshape(-1, key.shape[-2], key.shape[-1])
        packed_value = value.reshape(-1, value.shape[-2], value.shape[-1])
        packed_output = torch_xcpu.ops.scaled_dot_product_attention_varlen(
            packed_query,
            packed_key,
            packed_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            dropout_p=0.0,
            scale=self.scale,
            enable_gqa=self.num_heads > self.num_kv_heads,
        )
        output = packed_output.reshape(
            batch_size,
            query_length,
            self.num_heads,
            packed_output.shape[-1],
        )

        if was_flattened:
            output = output.reshape(batch_size, query_length, -1)
        return output
