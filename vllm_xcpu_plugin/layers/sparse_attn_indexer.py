# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillMetadata,
)


class XcpuSparseIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        parallel = vllm_config.parallel_config
        speculative = vllm_config.speculative_config
        compress_ratio = getattr(kv_cache_spec, "compress_ratio", 1)

        missing_capabilities = []
        if compress_ratio != 1:
            missing_capabilities.append(
                f"compress_ratio={compress_ratio} (compressed-cache execution)"
            )
        if parallel.decode_context_parallel_size != 1:
            missing_capabilities.append(
                f"DCP={parallel.decode_context_parallel_size} "
                "(local KV gather and global top-k merge)"
            )
        if parallel.prefill_context_parallel_size != 1:
            missing_capabilities.append(
                f"PCP={parallel.prefill_context_parallel_size} "
                "(K/slot-mapping all-gather)"
            )
        if speculative is not None:
            missing_capabilities.append(
                "speculative decode (multi-token decode and variable-length padding)"
            )

        errors = []
        if missing_capabilities:
            errors.append(
                "unsupported because required execution capabilities are missing: "
                + ", ".join(missing_capabilities)
            )
        if errors:
            raise NotImplementedError(
                "XCPU sparse indexer configuration rejected; " + "; ".join(errors)
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)


class XcpuSparseIndexerBackend(DeepseekV32IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "XCPU_SPARSE_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[XcpuSparseIndexerMetadataBuilder]:
        return XcpuSparseIndexerMetadataBuilder


def xcpu_indexer_prefill(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    paged_k_cache: torch.Tensor,
    metadata: DeepseekV32IndexerPrefillMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    import torch_xcpu

    workspace = None
    workspace_key = None
    for chunk in metadata.chunks:
        key = (chunk.block_table.data_ptr(), chunk.total_seq_lens)
        if not chunk.skip_kv_gather or workspace is None or workspace_key != key:
            workspace = torch_xcpu.ops.gather_paged_k_bf16(
                paged_k_cache,
                chunk.block_table.contiguous(),
                torch.diff(chunk.cu_seq_lens).contiguous(),
            )
            workspace_key = key
        token_slice = slice(chunk.token_start, chunk.token_end)
        torch_xcpu.ops.indexer_prefill_topk(
            q_quant[token_slice].contiguous(),
            weights[token_slice].float().contiguous(),
            workspace,
            chunk.cu_seqlen_ks.contiguous(),
            chunk.cu_seqlen_ke.contiguous(),
            output[token_slice].contiguous(),
        )
    return output


def xcpu_indexer_decode(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    paged_k_cache: torch.Tensor,
    metadata: DeepSeekV32IndexerDecodeMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    import torch_xcpu

    torch_xcpu.ops.indexer_decode_topk(
        q_quant.contiguous(),
        weights.float().contiguous(),
        paged_k_cache,
        metadata.block_table.contiguous(),
        metadata.seq_lens.contiguous(),
        output.contiguous(),
    )
    return output


# TODO: impl as xcpu op
@torch.library.custom_op(
    "vllm_xcpu_plugin::indexer_insert_k", mutates_args=["kv_cache"]
)
def indexer_insert_k(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    head_dim: int,
) -> None:
    # Runs as a fallback kernel: inductor hands over the runtime tensor
    # handles, so the (per-step varying) token count never enters the
    # compiled graph as a static size.
    slots_cpu = slot_mapping.flatten().cpu()
    valid_token_indices_cpu = torch.nonzero(slots_cpu >= 0).flatten()
    if valid_token_indices_cpu.numel() > 0:
        valid_token_indices = valid_token_indices_cpu.to(device=k.device)
        valid_slots = slots_cpu[valid_token_indices_cpu].to(device=k.device)
        valid_k = torch.index_select(k, 0, valid_token_indices)
        kv_cache.view(-1, head_dim).index_copy_(0, valid_slots, valid_k)


@indexer_insert_k.register_fake
def _(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    head_dim: int,
) -> None:
    return


@SparseAttnIndexer.register_oot
class XcpuSparseAttnIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        num_tokens = hidden_states.shape[0]
        output = self.topk_indices_buffer[:num_tokens]
        output.fill_(-1)
        metadata = get_forward_context().attn_metadata
        if metadata is None or self.k_cache.kv_cache.numel() == 0:
            return self.topk_indices_buffer
        layer_metadata = (
            metadata[self.k_cache.prefix] if isinstance(metadata, dict) else metadata
        )
        assert isinstance(layer_metadata, DeepseekV32IndexerMetadata)

        # Kept behind the custom-op boundary: tracing the nonzero/index_copy
        # chain would bake the compile-time token count into the graph as a
        # static size and break reuse at other batch sizes.
        indexer_insert_k(
            k,
            self.k_cache.kv_cache,
            layer_metadata.slot_mapping,
            self.k_cache.head_dim,
        )

        if layer_metadata.prefill is not None:
            xcpu_indexer_prefill(
                q_quant,
                weights,
                self.k_cache.kv_cache,
                layer_metadata.prefill,
                output,
            )
        if layer_metadata.decode is not None:
            if layer_metadata.prefill is None:
                # Pure decode: the decode segment covers every row; derive the
                # bound from the tensor shape so the compiled graph stays
                # batch-size dynamic (equals num_decode_tokens here, whose
                # metadata int is frozen at trace time).
                num_decode_tokens = q_quant.shape[0]
            else:
                num_decode_tokens = layer_metadata.num_decode_tokens
            xcpu_indexer_decode(
                q_quant[:num_decode_tokens],
                weights[:num_decode_tokens],
                self.k_cache.kv_cache,
                layer_metadata.decode,
                output[:num_decode_tokens],
            )
        return self.topk_indices_buffer
