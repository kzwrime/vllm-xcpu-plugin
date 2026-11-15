# SPDX-License-Identifier: Apache-2.0
"""Engram on XCPU: n-gram hash, table lookup and the gated stream injection.

The upstream layers are pluggable; these OOT subclasses keep their structure
(staging rows, TP/DP sharding) and replace only the three Triton kernels with
the XCPU ops. The table stays an ordinary device tensor (``cpu_offload=False``,
see the platform hook).

With Engram disaggregation (``VLLM_XCPU_ENABLE_ENGRAM_DISAGG``, a registered
``EngramClient``) the attention rank holds no table: the model forward sends
every forward's hash ids to the Engram ranks once (``gather_engram_hashes``
rebound in ``model.py``) and ``embed`` decodes the returned rows of its own
tokens, so the staging rows, the DP exchange and the TP all-gather are
bypassed.
"""

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
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

from vllm_xcpu_plugin.engram_disagg.attn.runtime import get_engram_client


def _xcpu_deepseek_v41_engram_embed(
    out: torch.Tensor, hash_ids: torch.Tensor, layer_name: LayerNameType
) -> None:
    """``Engram.embed`` behind an opaque boundary: the staged rows of a
    DP-sharded table are exchanged with a token slot read from the forward
    context's DP metadata, which a compiled graph must not bake in; with
    Engram disaggregation the rows arrive from the Engram ranks."""
    self = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    self._fill_rows(out, hash_ids)


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


def _reject_remote_engram_table(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    raise RuntimeError(
        "the Engram table is served by the Engram ranks; the attention rank's "
        "loader must skip engram.embed.* (see engram_disagg.attn.model_loader)"
    )


@ParallelEngramEmbedding.register_oot
class XcpuParallelEngramEmbedding(ParallelEngramEmbedding):
    def __init__(self, *args, **kwargs) -> None:
        if get_engram_client() is not None:
            # No table on this rank: zero-row parameters, and a loader that
            # refuses the checkpoint tensor should it ever reach the model.
            self._weight_loader = _reject_remote_engram_table
        super().__init__(*args, **kwargs)

    def _get_shard_info(self) -> tuple[int, int]:
        if get_engram_client() is not None:
            # The Engram ranks shard the columns; this rank sees all of them.
            return 1, 0
        return super()._get_shard_info()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if get_engram_client() is None:
            return super()._allocate_weights()
        return (
            torch.empty(0, self.dim, dtype=torch.float8_e4m3fn),
            torch.empty(0, self.dim // self.block_size, dtype=torch.uint8),
        )

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
        client = get_engram_client()
        if client is not None:
            client.bind_layer(self.use_sequence_parallel, self.layer_hash_index)

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        if get_engram_client() is None:
            super()._init_staging(max_tokens, head_dim)
            return
        # Rows are decoded straight into the per-forward output.
        self.staged_rows = None
        self._prefetch_stream = None

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        # Disaggregated: the model forward already sent this forward's request.
        if get_engram_client() is None:
            super().prepare_embeddings(hash_ids)

    def _fill_rows(self, out: torch.Tensor, hash_ids: torch.Tensor) -> None:
        """Fill ``out`` [local tokens, n_hash_cols, dim] with this layer's rows."""
        client = get_engram_client()
        if client is not None:
            client.fill_rows(self.layer_hash_index, out)
            return
        rows = self._ready_rows(hash_ids.shape[0])
        if self.embed_tokens.tp_size == 1:
            out.copy_(rows[:, : self.embed_tokens.n_hash_cols])
        elif self.use_sequence_parallel:
            import torch_xcpu

            tp_size = self.embed_tokens.tp_size
            num_tokens, local_heads, dim = rows.shape
            gathered = tensor_model_parallel_all_gather(rows, dim=0)
            chunk = (num_tokens + tp_size - 1) // tp_size
            torch_xcpu.ops.deepseek_v41_engram_select_rows(
                out,
                gathered,
                num_tokens,
                get_tensor_model_parallel_rank() * chunk,
                local_heads * dim,
            )
        else:
            rows = tensor_model_parallel_all_gather(rows, dim=1)
            out.copy_(rows[:, : self.embed_tokens.n_hash_cols])

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if get_engram_client() is None and not torch.compiler.is_compiling():
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
