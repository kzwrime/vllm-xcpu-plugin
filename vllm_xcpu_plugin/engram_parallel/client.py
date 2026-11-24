"""Model-facing client for EngramParallel (symmetric all-rank Engram sharding)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .layout import EngramParallelLayout
from .session import EngramParallelSession

logger = logging.getLogger(__name__)


class EngramParallelClient:
    """Implements the ``EngramClient`` protocol over ``EngramParallelSession``."""

    def __init__(
        self,
        session: EngramParallelSession,
        layout: EngramParallelLayout,
        *,
        dp_size: int | None = None,
        dp_rank: int | None = None,
    ) -> None:
        self._dp: tuple[int, int] | None = None
        if (dp_size is None) != (dp_rank is None):
            raise ValueError("dp_size and dp_rank must both be set or both omitted")
        if dp_size is not None and dp_rank is not None:
            if dp_size <= 0 or not (0 <= dp_rank < dp_size):
                raise ValueError(
                    f"invalid DP topology: dp_rank={dp_rank}, dp_size={dp_size}"
                )
            self._dp = (dp_size, dp_rank)
        self._session = session
        self._layout = layout
        self._tp: tuple[int, int] | None = None
        self._use_sequence_parallel: bool | None = None
        self._pending_layer_index: int | None = None
        self._bound_layers: set[int] = set()
        self._weights: dict[int, torch.Tensor] = {}
        self._scales: dict[int, torch.Tensor] = {}

        self._req_send: torch.Tensor | None = None
        self._req_recv: torch.Tensor | None = None
        self._reply_send: torch.Tensor | None = None
        self._reply_recv: torch.Tensor | None = None
        self._reply_index: torch.Tensor | None = None
        self._single_dp_tokens: torch.Tensor | None = None
        self._pending_tokens: int | None = None
        self._collected = False
        self._stopped = False

    @staticmethod
    def _check_eager(op: str) -> None:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                f"EngramParallelClient.{op} must run outside compiled graphs: "
                "the EngramParallel exchange ops are MPI side effects without "
                "fake impls; call them from eager code"
            )

    @property
    def session(self) -> EngramParallelSession:
        return self._session

    @property
    def layout(self) -> EngramParallelLayout:
        return self._layout

    def prepare_embedding(
        self, layout: Any, layer_hash_index: int
    ) -> tuple[int, int]:
        """Verify model layout against pre-built sharding and stage layer index."""
        resolved = EngramParallelLayout.from_engram_layout(
            layout, self._layout.world_size
        )
        if resolved != self._layout:
            raise RuntimeError(
                "model EngramLayout differs from EngramParallel bootstrap layout"
            )
        self._pending_layer_index = layer_hash_index
        return self._layout.layer_slice(layer_hash_index, self._session.world_rank)

    def consume_embedding_slice(
        self, num_embeddings: int, head_sizes: tuple[int, ...]
    ) -> tuple[int, int]:
        """Return ``(lo, hi)`` rows owned by this rank for the layer under init."""
        layer_index = self._pending_layer_index
        self._pending_layer_index = None
        if layer_index is None:
            raise RuntimeError(
                "consume_embedding_slice requires prepare_embedding first"
            )
        if (
            self._layout.num_embeddings[layer_index] != num_embeddings
            or self._layout.head_sizes[layer_index] != head_sizes
        ):
            raise RuntimeError(
                f"layer {layer_index} ParallelEngramEmbedding dimensions mismatch"
            )
        return self._layout.layer_slice(layer_index, self._session.world_rank)

    def initialize_from_engram_layout(self, layout: Any) -> None:
        self.initialize(
            EngramParallelLayout.from_engram_layout(
                layout, self._layout.world_size
            )
        )

    def bind_layer(
        self,
        use_sequence_parallel: bool,
        layer_hash_index: int,
        weight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        if self._req_send is not None:
            raise RuntimeError("engram layers must bind before the client initializes")
        if (
            self._use_sequence_parallel is not None
            and self._use_sequence_parallel != use_sequence_parallel
        ):
            raise RuntimeError("engram layers disagree on sequence parallelism")
        if layer_hash_index in self._bound_layers:
            raise RuntimeError(f"engram layer index {layer_hash_index} bound twice")
        if weight is None or scale is None:
            raise RuntimeError(
                "EngramParallel requires each layer's local table weight and scale"
            )
        self._use_sequence_parallel = use_sequence_parallel
        self._bound_layers.add(layer_hash_index)
        self._weights[layer_hash_index] = weight
        self._scales[layer_hash_index] = scale

    def _get_dp(self) -> tuple[int, int]:
        if self._dp is None:
            from vllm.distributed.parallel_state import (
                get_dp_group,
                model_parallel_is_initialized,
            )

            if model_parallel_is_initialized():
                dp_group = get_dp_group()
                self._dp = (dp_group.world_size, dp_group.rank_in_group)
            else:
                self._dp = (1, 0)
        return self._dp

    def _get_tp(self) -> tuple[int, int]:
        if self._tp is None:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.distributed.parallel_state import model_parallel_is_initialized

            if model_parallel_is_initialized():
                self._tp = (
                    get_tensor_model_parallel_world_size(),
                    get_tensor_model_parallel_rank(),
                )
            elif not self._use_sequence_parallel:
                dp_size, _ = self._get_dp()
                tp_size = max(1, self._layout.world_size // dp_size)
                self._tp = (tp_size, self._session.world_rank % tp_size)
            else:
                self._tp = (
                    get_tensor_model_parallel_world_size(),
                    get_tensor_model_parallel_rank(),
                )
        return self._tp

    def local_window(self, num_tokens: int) -> tuple[int, int]:
        """``(start, count)`` of this rank's tokens among ``num_tokens``."""
        if not self._use_sequence_parallel:
            return 0, num_tokens
        tp_size, tp_rank = self._get_tp()
        chunk = (num_tokens + tp_size - 1) // tp_size
        start = min(tp_rank * chunk, num_tokens)
        return start, min(chunk, num_tokens - start)

    def initialize(self, layout: EngramParallelLayout | None = None) -> None:
        if layout is not None and layout != self._layout:
            raise RuntimeError("EngramParallel layout mismatch during initialize")
        if self._bound_layers != set(range(self._layout.layers)):
            raise RuntimeError(
                f"bound engram layers {sorted(self._bound_layers)} do not cover the "
                f"{self._layout.layers} engram layers of the checkpoint"
            )
        if self._req_send is not None:
            return

        dp_size, dp_rank = self._get_dp()
        tp_size, tp_rank = self._get_tp()
        use_sp = bool(self._use_sequence_parallel)

        proposal = self._session.max_tokens_proposal
        if use_sp:
            proposal = (proposal + tp_size - 1) // tp_size

        weights = [self._weights[i] for i in range(self._layout.layers)]
        scales = [self._scales[i] for i in range(self._layout.layers)]
        max_tokens = self._session.initialize(
            self._layout,
            weights,
            scales,
            max_tokens_proposal=proposal,
            dp_size=dp_size,
            dp_rank=dp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_sequence_parallel=use_sp,
        )

        from torch_xcpu import ops as xcpu_ops

        self._stopped = False
        device = torch.device("cpu")
        slot_ints = xcpu_ops.deepseek_v41_engram_parallel_slot_ints(
            max_tokens, self._layout.max_cols_per_rank
        )
        slot_bytes = xcpu_ops.deepseek_v41_engram_parallel_slot_bytes(
            max_tokens, self._layout.max_cols_per_rank
        )
        world_size = self._layout.world_size
        self._req_send = torch.zeros(
            world_size, slot_ints, dtype=torch.int32, device=device
        )
        self._req_recv = torch.zeros(
            world_size, slot_ints, dtype=torch.int32, device=device
        )
        self._reply_send = torch.zeros(
            world_size, slot_bytes, dtype=torch.uint8, device=device
        )
        self._reply_recv = torch.zeros(
            world_size, slot_bytes, dtype=torch.uint8, device=device
        )
        self._reply_index = torch.zeros(
            max_tokens,
            self._layout.layers,
            self._layout.total_cols,
            dtype=torch.int32,
            device=device,
        )
        self._single_dp_tokens = torch.zeros(1, dtype=torch.int32, device=device)

    def _resolve_dp_tokens(self, total_tokens: int) -> torch.Tensor:
        dp_size, dp_rank = self._get_dp()
        if dp_size == 1:
            assert self._single_dp_tokens is not None
            self._single_dp_tokens[0] = total_tokens
            return self._single_dp_tokens

        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.v1.worker.dp_utils import should_skip_dp_coordination

        if should_skip_dp_coordination():
            raise RuntimeError(
                "EngramParallel does not support skip_dp_coordination when "
                "data_parallel_size > 1"
            )
        if not is_forward_context_available():
            raise RuntimeError(
                "EngramParallel requires an active ForwardContext with dp_metadata "
                "when data_parallel_size > 1"
            )
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError(
                "EngramParallel requires ForwardContext.dp_metadata when "
                "data_parallel_size > 1"
            )
        dp_tokens = dp_metadata.num_tokens_across_dp_cpu
        if dp_tokens.device.type != "cpu":
            dp_tokens = dp_tokens.cpu()
        if dp_tokens.dtype != torch.int32 or not dp_tokens.is_contiguous():
            dp_tokens = dp_tokens.to(dtype=torch.int32).contiguous()
        if dp_tokens.dim() != 1 or dp_tokens.shape[0] != dp_size:
            raise RuntimeError(
                f"EngramParallel dp_metadata.num_tokens_across_dp_cpu must have "
                f"shape ({dp_size},), got {tuple(dp_tokens.shape)}"
            )
        recorded = int(dp_tokens[dp_rank].item())
        if recorded != total_tokens:
            raise RuntimeError(
                f"EngramParallel local tokens {total_tokens} disagree with "
                f"dp_metadata.num_tokens_across_dp_cpu[{dp_rank}]={recorded}"
            )
        return dp_tokens

    def request(self, engram_hashes: torch.Tensor) -> None:
        self._check_eager("request")
        layout, max_tokens = self._session.require_initialized()
        assert (
            self._req_send is not None
            and self._req_recv is not None
            and self._reply_index is not None
        )
        if self._pending_tokens is not None and not self._collected:
            raise RuntimeError("a previous EngramParallel request was never collected")
        if engram_hashes.dim() != 3 or engram_hashes.shape[1] != layout.layers:
            raise ValueError(
                f"engram hashes must be [tokens, {layout.layers}, cols], got "
                f"{tuple(engram_hashes.shape)}"
            )
        dp_tokens = self._resolve_dp_tokens(engram_hashes.shape[0])
        start, count = self.local_window(engram_hashes.shape[0])
        if count > max_tokens:
            raise RuntimeError(
                f"{count} engram tokens exceed the negotiated capacity {max_tokens}"
            )
        from torch_xcpu import ops as xcpu_ops

        xcpu_ops.deepseek_v41_engram_parallel_request(
            engram_hashes,
            start,
            count,
            self._req_send,
            self._req_recv,
            self._reply_index,
            self._session.communicator_handle,
            dp_tokens,
        )
        self._pending_tokens = count
        self._collected = False

    def fill_rows(self, layer_hash_index: int, rows: torch.Tensor) -> None:
        layout, max_tokens = self._session.require_initialized()
        assert (
            self._req_send is not None
            and self._req_recv is not None
            and self._reply_send is not None
            and self._reply_recv is not None
            and self._reply_index is not None
        )
        if self._pending_tokens is None:
            raise RuntimeError(
                "EngramParallel rows were requested before any request was sent"
            )
        if rows.shape[0] < self._pending_tokens:
            raise RuntimeError(
                f"engram rows buffer holds {rows.shape[0]} tokens, the request had "
                f"{self._pending_tokens}"
            )
        from torch_xcpu import ops as xcpu_ops

        if layer_hash_index == 0:
            if self._collected:
                raise RuntimeError(
                    "the EngramParallel reply of this forward was already collected"
                )
            self._check_eager("collect")
            xcpu_ops.deepseek_v41_engram_parallel_collect(
                self._req_send,
                self._req_recv,
                self._reply_send,
                self._reply_recv,
                self._session.communicator_handle,
            )
            self._collected = True
        elif not self._collected:
            raise RuntimeError(
                "engram layer 0 must collect the reply before later layers"
            )

        max_q = max_tokens * layout.max_cols_per_rank
        xcpu_ops.deepseek_v41_engram_parallel_decode(
            rows,
            self._reply_recv,
            self._reply_index,
            layer_hash_index,
            self._pending_tokens,
            layout.layers,
            max_q,
        )

    def stop(self) -> None:
        """Release native state and duplicated communicator. Idempotent."""
        if self._stopped:
            return
        self._check_eager("stop")
        self._stopped = True
        if self._req_send is None or (
            self._pending_tokens is not None and not self._collected
        ):
            logger.error(
                "EngramParallelClient.stop() called with %s; aborting MPI "
                "communicator instead of freeing an uninitialized or "
                "in-flight exchange",
                "an uninitialized session"
                if self._req_send is None
                else f"an uncollected request (pending_tokens={self._pending_tokens})",
            )
            # Do not cleanup() native buffers or drop tensor references while
            # MPI_Ialltoall may still hold pointers; abort terminates ranks cleanly.
            self._session.abort(1)
            return

        from torch_xcpu import ops as xcpu_ops

        torch.accelerator.synchronize()
        xcpu_ops.deepseek_v41_engram_parallel_cleanup()
        self._session.close()
        self._req_send = None
        self._req_recv = None
        self._reply_send = None
        self._reply_recv = None
        self._reply_index = None
        self._single_dp_tokens = None
        self._pending_tokens = None
        self._collected = False
        self._weights.clear()
        self._scales.clear()
