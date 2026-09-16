"""Model-facing AF-EP client contract and process-local registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class ExpertsClient(Protocol):
    """Operations needed by the model, independent of the transport version.

    Each worker owns one client shared by its routed layers. Callers serialize
    forwards; concrete implementations own transport state and scratch buffers.
    No session, communicator or wire metadata is exposed through this contract.
    """

    def initialize(self, hidden_size: int, topk: int, dtype: torch.dtype) -> None:
        """Prepare transport after loading weights and before model execution."""
        ...

    def execute_layer(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
    ) -> torch.Tensor:
        """Return routed output, weighted and reduced across selected experts.

        The output matches the hidden-state shape, device and dtype. Shared
        experts remain with vLLM and are not executed by this call. Device work
        may be queued; the output must be usable on the caller's execution stream.
        """
        ...

    def sync_forward_entry(self) -> None:
        """Perform optional forward-entry synchronization, or return immediately."""
        ...


# vLLM's layer factory cannot receive a worker argument. This bridge supplies
# its client during model construction and optional ModelRunner synchronization.
_REMOTE_EXPERTS_CLIENT: ExpertsClient | None = None


def get_remote_experts_client() -> ExpertsClient | None:
    return _REMOTE_EXPERTS_CLIENT


def register_remote_experts_client(client: ExpertsClient) -> None:
    """Register one worker's client before constructing its model layers."""
    global _REMOTE_EXPERTS_CLIENT
    if _REMOTE_EXPERTS_CLIENT is not None:
        raise RuntimeError("an AF-EP attention worker is already registered")
    _REMOTE_EXPERTS_CLIENT = client


def unregister_remote_experts_client(client: ExpertsClient) -> None:
    """Detach a stopped worker's client without touching transport resources.

    Call only after the worker stops submitting forwards. Existing layers still
    own references; transport cleanup requires a separate coordinated A/F stop.
    Detaching does not make it safe to start a new transport session.
    """
    global _REMOTE_EXPERTS_CLIENT
    if _REMOTE_EXPERTS_CLIENT is not client:
        raise RuntimeError("cannot unregister another AF-EP worker's client")
    _REMOTE_EXPERTS_CLIENT = None
