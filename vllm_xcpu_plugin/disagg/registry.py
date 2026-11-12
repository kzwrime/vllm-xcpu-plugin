"""Process-local registration of one attention-side client per worker."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class ClientRegistry(Generic[T]):
    """vLLM's layer factory cannot receive a worker argument; this bridge
    supplies the worker's client during model construction and forward."""

    def __init__(self, what: str) -> None:
        self._what = what
        self._client: T | None = None

    def get(self) -> T | None:
        return self._client

    def require(self) -> T:
        if self._client is None:
            raise RuntimeError(f"no {self._what} client is registered")
        return self._client

    def register(self, client: T) -> None:
        """Register one worker's client before constructing its model layers."""
        if self._client is not None:
            raise RuntimeError(f"a {self._what} client is already registered")
        self._client = client

    def unregister(self, client: T) -> None:
        """Detach a stopped worker's client; transport cleanup is separate."""
        if self._client is not client:
            raise RuntimeError(
                f"cannot unregister another worker's {self._what} client"
            )
        self._client = None
