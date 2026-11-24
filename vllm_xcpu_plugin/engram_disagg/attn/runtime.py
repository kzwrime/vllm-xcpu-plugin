"""Model-facing Engram client contract and process-local registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vllm_xcpu_plugin.disagg import ClientRegistry

if TYPE_CHECKING:
    import torch


class EngramClient(Protocol):
    """What the model needs from the attention side of the exchange.

    One client per worker, shared by the engram layers. Calls are serialized
    by the model forward: ``request`` once per forward (before the decoder
    layers, on the full token stream), then ``fill_rows`` once per engram
    layer in layer order.
    """

    @property
    def session(self) -> Any: ...

    def prepare_embedding(
        self, layout: Any, layer_hash_index: int
    ) -> tuple[int, int] | None:
        """Stage a layer before ``ParallelEngramEmbedding`` construction."""
        ...

    def consume_embedding_slice(
        self, num_embeddings: int, head_sizes: tuple[int, ...]
    ) -> tuple[int, int] | None:
        """Return ``(lo, hi)`` rows owned locally, or ``None`` if remote."""
        ...

    def bind_layer(
        self,
        use_sequence_parallel: bool,
        layer_hash_index: int,
        weight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        """Register one engram layer's token sharding (and optional local slice)
        before initialize."""
        ...

    def initialize_from_engram_layout(self, layout: Any) -> None:
        """Negotiate the exchange from ``EngramLayout`` after weights load."""
        ...

    def request(self, engram_hashes: torch.Tensor) -> None:
        """Post this rank's tokens' hash ids ``[T, layers, cols]``."""
        ...

    def fill_rows(self, layer_hash_index: int, rows: torch.Tensor) -> None:
        """Write layer ``layer_hash_index``'s rows into ``rows`` [tokens, cols, dim]
        (layer 0 collects the reply first)."""
        ...

    def stop(self) -> None:
        """Tear down exchange state and communicator."""
        ...


# vLLM's layer factory cannot receive a worker argument. This bridge supplies
# the worker's client during model construction and forward.
REGISTRY: ClientRegistry[EngramClient] = ClientRegistry("Engram")


def get_engram_client() -> EngramClient | None:
    return REGISTRY.get()


def require_engram_client() -> EngramClient:
    return REGISTRY.require()


def register_engram_client(client: EngramClient) -> None:
    REGISTRY.register(client)


def unregister_engram_client(client: EngramClient) -> None:
    REGISTRY.unregister(client)
