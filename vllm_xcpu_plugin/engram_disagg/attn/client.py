"""Attention-rank client: one request/reply exchange per forward."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

from ..common import EngramSession, EngramShardLayout

logger = init_logger(__name__)


class RemoteEngramClient:
    """Fixed-slot request, exact-count reply, decode per engram layer.

    Not reentrant: the model forward calls ``request`` once (on the full
    token stream, before sequence-parallel sharding), which posts the
    request send and the reply receive as non-blocking collectives; then
    ``fill_rows`` for engram layer 0 (which waits for both) and the later
    layers in order, so the Engram ranks' gather and the transfer overlap
    the decoder layers before the first engram layer. The tokens this rank
    requests are the ones it feeds to ``wkv``: its sequence-parallel shard
    (``Engram._local_kv_tokens``) or all tokens.

    The exchange buffers are CPU tensors held for the client's lifetime:
    MPI fills the reply buffer outside any stream task.
    """

    def __init__(self, session: EngramSession) -> None:
        assert session.cluster_type == ClusterType.ATTN
        self._session = session
        self._tp: tuple[int, int] | None = None
        self._use_sequence_parallel: bool | None = None
        self._bound_layers: set[int] = set()
        self._slots: torch.Tensor | None = None
        self._recv: torch.Tensor | None = None
        self._pending_tokens: int | None = None
        self._collected = False
        self._stopped = False

    @staticmethod
    def _check_eager(op: str) -> None:
        # The exchange ops have no fake impls on purpose: they post / wait MPI
        # collectives as stream tasks and must be called from eager model code
        # (model.forward / _run_layers), never from a traced graph.
        if torch.compiler.is_compiling():
            raise RuntimeError(
                f"RemoteEngramClient.{op} must run outside compiled graphs: the "
                "Engram exchange ops are MPI side effects without fake "
                "impls; call them from eager code (see "
                "deepseek_v41.engram.remote)"
            )

    @property
    def session(self) -> EngramSession:
        return self._session

    def prepare_embedding(
        self, layout: Any, layer_hash_index: int
    ) -> tuple[int, int] | None:
        del layout, layer_hash_index
        return None

    def consume_embedding_slice(
        self, num_embeddings: int, head_sizes: tuple[int, ...]
    ) -> tuple[int, int] | None:
        del num_embeddings, head_sizes
        return None

    def initialize_from_engram_layout(self, layout: Any) -> None:
        self.initialize(
            EngramShardLayout.from_engram_layout(
                layout, self._session.num_engram_ranks
            )
        )

    def bind_layer(
        self,
        use_sequence_parallel: bool,
        layer_hash_index: int,
        weight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        del weight, scale
        if self._slots is not None:
            raise RuntimeError("engram layers must bind before the client initializes")
        if (
            self._use_sequence_parallel is not None
            and self._use_sequence_parallel != use_sequence_parallel
        ):
            raise RuntimeError("engram layers disagree on sequence parallelism")
        if layer_hash_index in self._bound_layers:
            raise RuntimeError(f"engram layer index {layer_hash_index} bound twice")
        self._use_sequence_parallel = use_sequence_parallel
        self._bound_layers.add(layer_hash_index)

    def initialize(self, shard: EngramShardLayout) -> None:
        if self._bound_layers != set(range(shard.layers)):
            raise RuntimeError(
                f"bound engram layers {sorted(self._bound_layers)} do not cover the "
                f"{shard.layers} engram layers of the checkpoint"
            )
        from torch_xcpu import ops as xcpu_ops

        max_tokens = self._session.initialize(shard)
        if self._slots is not None:
            return
        self._stopped = False  # a renegotiated session can stop again
        device = torch.device("cpu")
        self._slots = torch.zeros(
            shard.num_engram_ranks,
            xcpu_ops.deepseek_v41_engram_remote_slot_ints(
                max_tokens, shard.layers, shard.cols_per_rank
            ),
            dtype=torch.int32,
            device=device,
        )
        self._recv = torch.zeros(
            shard.num_engram_ranks,
            xcpu_ops.deepseek_v41_engram_remote_slot_bytes(
                max_tokens, shard.layers, shard.cols_per_rank
            ),
            dtype=torch.uint8,
            device=device,
        )

    def local_window(self, num_tokens: int) -> tuple[int, int]:
        """``(start, count)`` of this rank's tokens among ``num_tokens``:
        the sequence-parallel shard of ``ceil(T / tp)`` tokens (the last one
        may be short or empty) or every token."""
        if not self._use_sequence_parallel:
            return 0, num_tokens
        if self._tp is None:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            self._tp = (
                get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank(),
            )
        tp_size, tp_rank = self._tp
        chunk = (num_tokens + tp_size - 1) // tp_size
        start = min(tp_rank * chunk, num_tokens)
        return start, min(chunk, num_tokens - start)

    def request(self, engram_hashes: torch.Tensor) -> None:
        self._check_eager("request")
        shard, max_tokens = self._session.require_initialized()
        assert self._slots is not None
        if self._pending_tokens is not None and not self._collected:
            raise RuntimeError("a previous Engram request was never collected")
        if engram_hashes.dim() != 3 or engram_hashes.shape[1] != shard.layers:
            raise ValueError(
                f"engram hashes must be [tokens, {shard.layers}, cols], got "
                f"{tuple(engram_hashes.shape)}"
            )
        start, count = self.local_window(engram_hashes.shape[0])
        if count > max_tokens:
            raise RuntimeError(
                f"{count} engram tokens exceed the negotiated capacity {max_tokens}"
            )
        from torch_xcpu import ops as xcpu_ops

        xcpu_ops.deepseek_v41_engram_remote_request(
            engram_hashes,
            start,
            count,
            self._slots,
            self._recv,
            self._session.metadata,
            self._session.communicator_handle,
        )
        self._pending_tokens = count
        self._collected = False

    def fill_rows(self, layer_hash_index: int, rows: torch.Tensor) -> None:
        shard, _ = self._session.require_initialized()
        assert self._recv is not None
        if self._pending_tokens is None:
            raise RuntimeError("Engram rows were requested before any request was sent")
        if rows.shape[0] < self._pending_tokens:
            raise RuntimeError(
                f"engram rows buffer holds {rows.shape[0]} tokens, the request had "
                f"{self._pending_tokens}"
            )
        from torch_xcpu import ops as xcpu_ops

        if layer_hash_index == 0:
            if self._collected:
                raise RuntimeError(
                    "the Engram reply of this forward was already collected"
                )
            self._check_eager("collect")
            xcpu_ops.deepseek_v41_engram_remote_collect(
                self._recv, self._session.metadata, self._session.communicator_handle
            )
            self._collected = True
        elif not self._collected:
            raise RuntimeError(
                "engram layer 0 must collect the reply before later layers"
            )
        xcpu_ops.deepseek_v41_engram_remote_decode(
            rows,
            self._recv,
            layer_hash_index,
            self._pending_tokens,
            shard.layers,
            shard.cols_per_rank,
        )

    def stop(self) -> None:
        """Tell the Engram ranks to exit. Idempotent.

        Without a completed ``initialize`` no stop request can be formed (the
        capacity was never negotiated), and the Engram ranks — and possibly
        the other attention ranks — are blocked in the initialization
        collective, so abort the MPI world to tear them down now instead of
        leaving them to the launcher."""
        if self._stopped:
            return
        self._stopped = True
        if self._slots is None:
            logger.error(
                "Engram disaggregation client stopping before initialize: "
                "the Engram ranks are blocked in the initialization "
                "collective; aborting the MPI world."
            )
            self._session.pair_comm.Abort(1)
            return
        self._check_eager("stop")
        from torch_xcpu import ops as xcpu_ops

        torch.accelerator.synchronize()
        xcpu_ops.deepseek_v41_engram_remote_stop(
            self._slots, self._session.metadata, self._session.communicator_handle
        )
        torch.accelerator.synchronize()
        xcpu_ops.deepseek_v41_engram_remote_cleanup()
        # The native context is gone; drop the negotiated cache with it so a
        # later initialize() renegotiates instead of serving stale state.
        self._session.finalize()
        self._slots = None
        self._recv = None
