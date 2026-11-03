# SPDX-License-Identifier: Apache-2.0
"""Engram on XCPU: n-gram hash, table lookup and the gated stream injection.

The upstream layers are pluggable; these OOT subclasses keep their structure
(staging rows, TP/DP sharding) and replace only the three Triton kernels with
the XCPU ops. The table stays an ordinary device tensor (``cpu_offload=False``,
see the platform hook).
"""

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    Engram,
    NgramHashState,
    ParallelEngramEmbedding,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)


def _xcpu_deepseek_v41_engram_embed(
    out: torch.Tensor, hash_ids: torch.Tensor, layer_name: LayerNameType
) -> None:
    """``Engram.embed`` behind an opaque boundary: the staged rows of a
    DP-sharded table are exchanged with a token slot read from the forward
    context's DP metadata, which a compiled graph must not bake in."""
    self = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    out.copy_(Engram.embed(self, hash_ids))


def _xcpu_deepseek_v41_engram_embed_fake(
    out: torch.Tensor, hash_ids: torch.Tensor, layer_name: LayerNameType
) -> None:
    return None


direct_register_custom_op(
    op_name="xcpu_deepseek_v41_engram_embed",
    op_func=_xcpu_deepseek_v41_engram_embed,
    mutates_args=["out"],
    fake_impl=_xcpu_deepseek_v41_engram_embed_fake,
)


@NgramHashState.register_oot
class XcpuNgramHashState(NgramHashState):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        import torch_xcpu

        # Reference: vLLM
        # vllm/models/deepseek_v41/common/engram.py::NgramHashState.forward
        cache = None
        if self.use_slot_cache:
            cache = self._cache
            assert cache is not None and slot_mapping is not None
            assert block_table is not None
        return torch_xcpu.ops.deepseek_v41_engram_hash(
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
            self.pad_id,
            DEAD_ID,
            cache=cache,
            slot_mapping=slot_mapping if cache is not None else None,
            block_table=block_table if cache is not None else None,
            cache_block_size=self.block_size if cache is not None else 0,
        )


@ParallelEngramEmbedding.register_oot
class XcpuParallelEngramEmbedding(ParallelEngramEmbedding):
    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        import torch_xcpu

        # Reference: vLLM
        # vllm/models/deepseek_v41/common/engram.py::ParallelEngramEmbedding.lookup
        if indices.shape[0] == 0:
            return
        weight, scales = self._storage()
        torch_xcpu.ops.deepseek_v41_engram_lookup(
            weight,
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            self.head_start,
            self.n_hash_cols,
        )


@Engram.register_oot
class XcpuEngram(Engram):
    def __init__(self, *args, prefix: str, **kwargs) -> None:
        super().__init__(*args, prefix=prefix, **kwargs)
        self.layer_name = prefix
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            return super().embed(hash_ids)
        num_tokens = hash_ids.shape[0]
        if self.use_sequence_parallel:
            num_tokens = (num_tokens + self.embed_tokens.tp_size - 1) // (
                self.embed_tokens.tp_size
            )
        rows = hash_ids.new_empty(
            (num_tokens, self.embed_tokens.n_hash_cols, self.embed_tokens.dim),
            dtype=torch.bfloat16,
        )
        torch.ops.vllm.xcpu_deepseek_v41_engram_embed(
            rows, hash_ids, _encode_layer_name(self.layer_name)
        )
        return rows

    def _post_wkv(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor,
        token_mask: torch.Tensor | None,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        import torch_xcpu

        # Reference: vLLM
        # vllm/models/deepseek_v41/common/engram.py::Engram._post_wkv
        # The upstream asserts relate symbolic token counts, which a traced
        # graph would guard on; the op's own checks (TORCH_XCPU_ENABLE_CHECK)
        # cover the shapes.
        return torch_xcpu.ops.deepseek_v41_engram_post_wkv(
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            token_mask,
            num_kv_tokens,
            self.eps,
            self.clamp_value,
        )
