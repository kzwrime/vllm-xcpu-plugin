"""Model-facing Engram client contract and process-local registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from vllm_xcpu_plugin.disagg import ClientRegistry

if TYPE_CHECKING:
    import torch

    from ..common import EngramSession, EngramShardLayout


class EngramClient(Protocol):
    """What the model needs from the attention side of the exchange.

    One client per worker, shared by the engram layers. Calls are serialized
    by the model forward: ``request`` once per forward (before the decoder
    layers, on the full token stream), then ``fill_rows`` once per engram
    layer in layer order.
    """

    # The transport session (worker bootstrap logging, shard rank counts).
    @property
    def session(self) -> EngramSession: ...

    def bind_layer(self, use_sequence_parallel: bool, layer_hash_index: int) -> None:
        """Register one engram layer's token sharding before initialize."""
        ...

    def initialize(self, shard: EngramShardLayout) -> None:
        """Negotiate the exchange after the weights are loaded."""
        ...

    def request(self, engram_hashes: torch.Tensor) -> None:
        """Post this rank's tokens' hash ids ``[T, layers, cols]`` to the
        Engram ranks and the receive of their reply."""
        ...

    def fill_rows(self, layer_hash_index: int, rows: torch.Tensor) -> None:
        """Write layer ``layer_hash_index``'s rows into ``rows`` [tokens, cols, dim]
        (layer 0 waits for the reply first)."""
        ...

    def stop(self) -> None:
        """Tell the Engram ranks to exit (every attention rank must call it)."""
        ...


# vLLM's layer factory cannot receive a worker argument. This bridge supplies
# the worker's client during model construction and forward.
REGISTRY: ClientRegistry[EngramClient] = ClientRegistry("Engram disaggregation")


def get_engram_client() -> EngramClient | None:
    return REGISTRY.get()


def require_engram_client() -> EngramClient:
    return REGISTRY.require()


def register_engram_client(client: EngramClient) -> None:
    REGISTRY.register(client)


def unregister_engram_client(client: EngramClient) -> None:
    REGISTRY.unregister(client)
