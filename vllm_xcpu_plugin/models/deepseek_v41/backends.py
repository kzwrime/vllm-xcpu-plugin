# SPDX-License-Identifier: Apache-2.0
"""XCPU attention backends and metadata builders for DeepSeek-V4.1.

The upstream builders are reused wherever they are platform neutral; their
Triton metadata kernels run through the Fake Triton adapters in
``triton_adapters``. What changes:

* SWA: the sparse MLA step metadata (``deepseek_v41_sparse_mla_metadata``)
  per layer type and call (decode rows, prefill rows) instead of the FlashMLA
  tile-scheduler plans; every same-type layer of the step reuses it.
* Compressed KV: same metadata as ``FLASHMLA_SPARSE_DSV41``.
* Indexer: a per-token paged indexer over the slot mapping (the upstream
  compressed slot mapping for ratio 2), block table and
  the token -> request map, plus the step schedule
  (``deepseek_v41_indexer_metadata``) of each row chunk that fits the logits
  budget (``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB``, as the DeepGEMM prefill
  chunking). Candidate consumers build their metadata once per step, after
  the candidate source layer wrote the blocks.
"""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

# Block size, in tokens, of the compressed-KV and indexer pages. Both page
# kinds hold block_size // compress_ratio entries.
XCPU_DSV41_BLOCK_SIZE = 128


class XcpuDeepseekV41SWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """The upstream SWA metadata plus the sparse MLA step metadata of each
    layer type and call, keyed ``(compress_ratio, "decode" | "prefill")``
    in ``metadata.xcpu_sparse_mla``."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    supports_draft_decode_metadata_update = False
    _step_cache: ClassVar[
        dict[torch.device, tuple[tuple[Any, ...], Any, int, set[int]]]
    ] = {}
    decode_swa_indices_noncausal: torch.Tensor | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hf_config = self.vllm_config.model_config.hf_config
        self.num_local_heads = (
            hf_config.num_attention_heads // get_tensor_model_parallel_world_size()
        )
        compress_ratios = getattr(hf_config, "compress_ratios", None) or [1]
        # Compressed index width per layer type: the indexer top-k (ratio 1/2).
        self.compressed_widths = {
            int(r): 0 if int(r) == 0 else hf_config.index_topk
            for r in set(compress_ratios)
        }
        # The widest SWA list of each call: the window (or DSpark's
        # non-causal width) for decode, window + image span for prefill.
        self.max_swa_widths = {
            "decode": max(self.window_size, self.noncausal_index_width),
            "prefill": self.prefill_index_width,
        }
        self._xcpu_buffers: dict[tuple[int, str], tuple[torch.Tensor, ...]] = {}

    def build_tile_scheduler(self, num_decode_tokens: int) -> dict:
        return dict.fromkeys(super().build_tile_scheduler(0))

    def _sparse_mla_metadata(self, key: tuple[int, str], tokens: int, max_kv: int):
        import torch_xcpu

        buffers = self._xcpu_buffers.get(key)
        if buffers is None:
            # Sized for the token budget and this call's widest list.
            ratio, part = key
            buffers = torch_xcpu.ops.deepseek_v41_sparse_mla_allocate_metadata(
                self.max_num_batched_tokens,
                self.num_local_heads,
                self.max_swa_widths[part] + self.compressed_widths[ratio],
                self.device,
            )
            self._xcpu_buffers[key] = buffers
        return torch_xcpu.ops.deepseek_v41_sparse_mla_metadata(
            tokens, self.num_local_heads, max_kv, self.device, buffers=buffers
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ):
        cm = common_attn_metadata
        has_mm = (
            self.max_image_tokens > 0
            and bool(cm.mm_req_doc_ranges)
            and any(cm.mm_req_doc_ranges.values())
        )
        cache_key = (
            self.block_size,
            self.window_size,
            self.prefill_index_width,
            self.noncausal_index_width,
            self.decode_threshold,
            self.num_local_heads,
            common_prefix_len,
            fast_build,
            cm.query_start_loc.data_ptr(),
            cm.query_start_loc._version,
            cm.seq_lens.data_ptr(),
            cm.seq_lens._version,
            cm.num_reqs,
            cm.num_actual_tokens,
            cm.max_query_len,
            cm.causal,
            None
            if replay_start is None
            else (replay_start.data_ptr(), replay_start._version),
        )
        cached = self._step_cache.get(self.device)
        if not has_mm and cached is not None:
            cached_key, base, base_slot_version, seen = cached
            if cached_key == cache_key and id(self) not in seen:
                import torch_xcpu

                seen.add(id(self))
                num_decode_tokens = base.num_decode_tokens
                num_prefill_tokens = base.num_prefill_tokens
                same_slots = (
                    cm.slot_mapping.data_ptr() == base.slot_mapping.data_ptr()
                    and cm.slot_mapping._version == base_slot_version
                    and cm.slot_mapping.shape == base.slot_mapping.shape
                )
                if same_slots:
                    is_valid_token = base.is_valid_token
                else:
                    is_valid_token = self.is_valid_token[: cm.slot_mapping.shape[0]]
                    torch.ge(cm.slot_mapping, 0, out=is_valid_token)
                eff_replay = (
                    self.no_replay_start if replay_start is None else replay_start
                )
                non_causal = not cm.causal
                decode_buf = self.decode_swa_indices
                if num_decode_tokens > 0:
                    if non_causal:
                        if self.decode_swa_indices_noncausal is None:
                            self.decode_swa_indices_noncausal = torch.zeros(
                                self._max_tokens,
                                1,
                                self.noncausal_index_width,
                                dtype=torch.int32,
                                device=self.device,
                            )
                        decode_buf = self.decode_swa_indices_noncausal
                    torch_xcpu.ops.deepseek_v41_compute_swa_metadata(
                        decode_buf,
                        None if same_slots else self.decode_swa_lens,
                        cm.query_start_loc,
                        cm.seq_lens,
                        base.token_to_req_indices,
                        is_valid_token,
                        cm.block_table_tensor,
                        None if non_causal else eff_replay,
                        decode_buf.stride(0),
                        self.window_size,
                        decode_buf.shape[-1],
                        self.block_size,
                        num_decode_tokens,
                        0,
                        non_causal,
                    )
                decode_swa_lens = (
                    base.decode_swa_lens
                    if same_slots
                    else self.decode_swa_lens[:num_decode_tokens]
                )
                prefill_swa_indices = None
                prefill_swa_lens = None
                if num_prefill_tokens > 0:
                    prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
                    prefill_swa_lens = (
                        base.prefill_swa_lens
                        if same_slots
                        else self.prefill_swa_lens[:num_prefill_tokens]
                    )
                    torch_xcpu.ops.deepseek_v41_compute_swa_metadata(
                        prefill_swa_indices,
                        None if same_slots else prefill_swa_lens,
                        cm.query_start_loc,
                        cm.seq_lens,
                        base.token_to_req_indices,
                        is_valid_token,
                        cm.block_table_tensor,
                        eff_replay,
                        prefill_swa_indices.stride(0),
                        self.window_size,
                        self.prefill_index_width,
                        self.block_size,
                        num_prefill_tokens,
                        num_decode_tokens,
                        False,
                    )
                metadata = replace(
                    base,
                    block_table=cm.block_table_tensor,
                    slot_mapping=cm.slot_mapping,
                    is_valid_token=is_valid_token,
                    decode_swa_indices=decode_buf[:num_decode_tokens],
                    decode_swa_lens=decode_swa_lens,
                    prefill_swa_indices=prefill_swa_indices,
                    prefill_swa_lens=prefill_swa_lens,
                )
                metadata.xcpu_sparse_mla = base.xcpu_sparse_mla
                return metadata

        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build, replay_start
        )
        calls = []
        if metadata.num_decode_tokens > 0:
            calls.append((
                "decode",
                metadata.num_decode_tokens,
                metadata.decode_swa_indices.shape[-1],
            ))
        if metadata.num_prefill_tokens > 0:
            calls.append((
                "prefill",
                metadata.num_prefill_tokens,
                metadata.prefill_swa_indices.shape[-1],
            ))
        metadata.xcpu_sparse_mla = {
            (ratio, part): self._sparse_mla_metadata(
                (ratio, part), tokens, swa_width + width
            )
            for ratio, width in self.compressed_widths.items()
            for part, tokens, swa_width in calls
        }
        if not has_mm:
            self._step_cache[self.device] = (
                cache_key,
                metadata,
                cm.slot_mapping._version,
                {id(self)},
            )
        return metadata


class XcpuDeepseekV41SWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_name() -> str:
        return "XCPU_DSV41_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[XcpuDeepseekV41SWAMetadataBuilder]:
        return XcpuDeepseekV41SWAMetadataBuilder


class XcpuDeepseekV41SparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER


class XcpuDeepseekV41SparseMLABackend(DeepseekV4SparseMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "XCPU_DSV41_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [XCPU_DSV41_BLOCK_SIZE]

    @staticmethod
    def get_builder_cls() -> type[XcpuDeepseekV41SparseMLAMetadataBuilder]:
        return XcpuDeepseekV41SparseMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


@dataclass
class XcpuDeepseekV41IndexerMetadata(AttentionMetadata):
    # [num_tokens] slots of the indexer K entry each token writes (-1 = none).
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    token_to_req_indices: torch.Tensor
    positions: torch.Tensor
    max_seq_len: int
    num_decode_tokens: int
    num_prefill_tokens: int
    # Row chunks [start, end) of the step with their dense-row schedules.
    chunks: list[tuple[int, int, torch.Tensor]]
    # Candidate consumers' (sparse_blocks, sparse_columns, schedule) per
    # chunk, built by the first consumer of the step.
    candidate_metadata: dict[int, Any] = field(default_factory=dict)


class XcpuDeepseekV41IndexerMetadataBuilder(
    AttentionMetadataBuilder[XcpuDeepseekV41IndexerMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        assert isinstance(kv_cache_spec.tokens_per_state, int)
        if kv_cache_spec.block_size != XCPU_DSV41_BLOCK_SIZE:
            raise ValueError(
                f"XCPU DeepSeek-V4.1 indexer pages hold {XCPU_DSV41_BLOCK_SIZE} "
                f"tokens; got block_size={kv_cache_spec.block_size}"
            )
        self.compress_ratio = kv_cache_spec.tokens_per_state
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.zeros(
            max_tokens, dtype=torch.int32, device=device
        )
        self.compressed_slot_mapping_buffer = torch.zeros(
            max_tokens, dtype=torch.int64, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XcpuDeepseekV41IndexerMetadata:
        import torch_xcpu

        cm = common_attn_metadata
        assert cm.positions is not None
        token_to_req = cm.token_to_req_indices(self.token_to_req_buffer)
        (_, _, num_decode_tokens, num_prefill_tokens) = split_decodes_and_prefills(
            cm, decode_threshold=self.reorder_batch_threshold or 1
        )
        num_tokens = num_decode_tokens + num_prefill_tokens
        positions = cm.positions[:num_tokens]
        # The [rows, columns] FP32 logits of one indexer call fit the budget.
        columns = torch_xcpu.ops.deepseek_v41_indexer_max_columns(
            cm.max_seq_len, self.compress_ratio
        )
        row_bytes = 4 * torch_xcpu.ops.deepseek_v41_indexer_logits_stride(columns)
        rows = max(1, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // row_bytes)
        chunks = [
            (
                start,
                min(start + rows, num_tokens),
                torch_xcpu.ops.deepseek_v41_indexer_metadata(
                    token_to_req[start : start + rows],
                    positions[start : start + rows],
                    self.compress_ratio,
                    cm.max_seq_len,
                ),
            )
            for start in range(0, num_tokens, rows)
        ]
        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            # As DeepseekV32IndexerMetadataBuilder.build (indexer.py) does.
            slot_mapping = get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.slot_mapping,
                cm.query_start_loc,
                cm.seq_lens,
                cm.block_table_tensor,
                self.kv_cache_spec.num_states,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
        return XcpuDeepseekV41IndexerMetadata(
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            token_to_req_indices=token_to_req,
            positions=cm.positions,
            max_seq_len=cm.max_seq_len,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            chunks=chunks,
        )


class XcpuDeepseekV41IndexerBackend(DeepseekV41IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "XCPU_DSV41_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [XCPU_DSV41_BLOCK_SIZE]

    @staticmethod
    def get_builder_cls() -> type[XcpuDeepseekV41IndexerMetadataBuilder]:
        return XcpuDeepseekV41IndexerMetadataBuilder


class XcpuDeepseekV41IndexerCache(DeepseekV4IndexerCache):
    """The upstream indexer K cache, served by the XCPU indexer backend."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        return XcpuDeepseekV41IndexerBackend
