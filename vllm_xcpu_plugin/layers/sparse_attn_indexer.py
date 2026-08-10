# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

_INDEXER_KV_CHUNK_SIZE = 512


@dataclass
class XcpuSparseIndexerMetadata(AttentionMetadata):
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int


class XcpuSparseIndexerMetadataBuilder(
    AttentionMetadataBuilder[XcpuSparseIndexerMetadata]
):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XcpuSparseIndexerMetadata:
        return XcpuSparseIndexerMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=common_attn_metadata.seq_lens.cpu(),
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
        )


class XcpuSparseIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "XCPU_SPARSE_INDEXER"

    @staticmethod
    def get_builder_cls(  # type: ignore[override]
    ) -> type[XcpuSparseIndexerMetadataBuilder]:
        return XcpuSparseIndexerMetadataBuilder


def _paged_cache_indices(
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    logical = torch.arange(seq_len, dtype=torch.long)
    block_table_cpu = block_table_row.cpu().to(torch.long)
    physical = block_table_cpu[logical // block_size] * block_size
    physical += logical % block_size
    return physical.to(device=device)


def xcpu_sparse_indexer_topk(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    paged_k_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_size: int,
    topk_tokens: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Reference-correct paged DSA indexer for arbitrary context lengths."""
    q = q_quant.to(torch.bfloat16)
    query_starts = query_start_loc_cpu.tolist()
    sequence_lengths = seq_lens_cpu.tolist()
    flat_cache = paged_k_cache.view(-1, paged_k_cache.shape[-1])
    output.fill_(-1)

    for request_idx, seq_len_value in enumerate(sequence_lengths):
        token_start = int(query_starts[request_idx])
        token_end = int(query_starts[request_idx + 1])
        query_len = token_end - token_start
        if query_len == 0:
            continue

        seq_len = int(seq_len_value)
        context_len = seq_len - query_len
        if seq_len <= topk_tokens:
            for local_query_idx in range(query_len):
                valid_len = context_len + local_query_idx + 1
                output[token_start + local_query_idx, :valid_len].copy_(
                    torch.arange(valid_len, dtype=torch.int32, device=output.device)
                )
            continue

        physical = _paged_cache_indices(
            block_table[request_idx], seq_len, block_size, paged_k_cache.device
        )
        keys = torch.index_select(flat_cache, 0, physical)
        request_q = q[token_start:token_end]
        request_weights = weights[token_start:token_end].float()
        valid_ends = (
            torch.arange(query_len, device=q.device, dtype=torch.long)
            + context_len
            + 1
        )

        candidate_values = None
        candidate_indices = None
        for key_start in range(0, seq_len, _INDEXER_KV_CHUNK_SIZE):
            key_end = min(seq_len, key_start + _INDEXER_KV_CHUNK_SIZE)
            key_chunk = keys[key_start:key_end]
            scores = torch.matmul(request_q, key_chunk.transpose(0, 1))
            logits = (
                torch.clamp(scores, min=0).float()
                * request_weights.unsqueeze(-1)
            ).sum(dim=1)
            logical = torch.arange(
                key_start, key_end, device=q.device, dtype=torch.long
            )
            logits.masked_fill_(
                logical.unsqueeze(0) >= valid_ends.unsqueeze(1), -torch.inf
            )

            local_k = min(topk_tokens, key_end - key_start)
            local_values, local_offsets = torch.topk(logits, local_k, dim=-1)
            local_indices = local_offsets + key_start
            if candidate_values is not None:
                assert candidate_indices is not None
                local_values = torch.cat((candidate_values, local_values), dim=-1)
                local_indices = torch.cat((candidate_indices, local_indices), dim=-1)
            keep = min(topk_tokens, local_values.shape[-1])
            candidate_values, selected = torch.topk(local_values, keep, dim=-1)
            candidate_indices = torch.gather(local_indices, 1, selected)

        assert candidate_values is not None and candidate_indices is not None
        candidate_indices = torch.where(
            candidate_values == -torch.inf,
            torch.full_like(candidate_indices, -1),
            candidate_indices,
        )
        output[token_start:token_end, : candidate_indices.shape[1]].copy_(
            candidate_indices
        )

    return output


@SparseAttnIndexer.register_oot
class XcpuSparseAttnIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # XCPU keeps indexer K in BF16. Q remains serialized FP8 and its scale
        # is already folded into ``weights`` by the generic GLM indexer.
        self.k_cache.head_dim = self.head_dim
        self.k_cache.dtype = torch.bfloat16
        self.k_cache.get_attn_backend = lambda: XcpuSparseIndexerBackend

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(q_quant, tuple):
            raise NotImplementedError("XCPU sparse indexer does not support FP4 Q")

        forward_context = get_forward_context()
        metadata = forward_context.attn_metadata
        if metadata is None or self.k_cache.kv_cache.numel() == 0:
            self.topk_indices_buffer[: hidden_states.shape[0]].fill_(-1)
            return self.topk_indices_buffer
        layer_metadata = (
            metadata[self.k_cache.prefix] if isinstance(metadata, dict) else metadata
        )
        assert isinstance(layer_metadata, XcpuSparseIndexerMetadata)

        slots_cpu = layer_metadata.slot_mapping.flatten().cpu()
        valid_token_indices_cpu = torch.nonzero(slots_cpu >= 0).flatten()
        if valid_token_indices_cpu.numel() > 0:
            valid_token_indices = valid_token_indices_cpu.to(device=k.device)
            valid_slots = slots_cpu[valid_token_indices_cpu].to(device=k.device)
            valid_k = torch.index_select(k, 0, valid_token_indices)
            self.k_cache.kv_cache.view(-1, self.head_dim).index_copy_(
                0, valid_slots, valid_k
            )
        return xcpu_sparse_indexer_topk(
            q_quant,
            weights,
            self.k_cache.kv_cache,
            layer_metadata.block_table,
            layer_metadata.query_start_loc_cpu,
            layer_metadata.seq_lens_cpu,
            layer_metadata.block_size,
            self.topk_tokens,
            self.topk_indices_buffer[: hidden_states.shape[0]],
        )
