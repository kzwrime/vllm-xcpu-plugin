"""vLLM sleep-mode backend for the xcpu platform.

``/sleep`` and ``/wake_up`` let an upper-level router hand the accelerator to
another job (inspection, borrowing) and take it back. The router guarantees
that no request reaches the instance in between, and vLLM itself never sleeps
or wakes on its own: both transitions are explicit HTTP calls.

On xcpu the whole of device memory stays mapped across the interval, so the
backend neither offloads weights nor discards the KV cache. It only releases
and reacquires the device compute runtime through torch_xcpu, whose hooks the
NPU port implements (``torch_xcpu_impl/include/device_runtime.h``). vLLM's
dispatch (scheduler pause, prefix-cache reset, executor sleeping tags) stays
untouched, so ``/is_sleeping`` and request queueing behave as upstream.
"""

from vllm.device_allocator.sleep_mode_backend import (
    SleepModeBackend,
    SleepModeBackendFactory,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

BACKEND_NAME = "xcpu"


class XcpuSleepModeBackend(SleepModeBackend):
    """Release / reacquire the device compute runtime; memory stays in place."""

    def suspend(self, level: int = 1) -> None:
        import torch_xcpu

        if torch_xcpu.is_device_runtime_sleeping():
            logger.warning("xcpu device runtime is already asleep; ignoring sleep")
            self._state = "SUSPENDED"
            return
        torch_xcpu.device_runtime_sleep(level)
        self._state = "SUSPENDED"
        logger.info("xcpu device runtime released (sleep level %d)", level)

    def resume(self, tags: list[str] | None = None) -> None:
        import torch_xcpu

        # The runtime is released as a whole, so a tagged partial wake-up
        # (e.g. ["weights"] then ["kv_cache"]) reacquires it on the first call
        # and is a no-op afterwards; the executor still tracks the tags.
        if not torch_xcpu.is_device_runtime_sleeping():
            logger.info("xcpu device runtime is already awake (tags=%s)", tags)
            self._state = "RUNNING"
            return
        self._state = "RESUMING"
        try:
            torch_xcpu.device_runtime_wake_up()
        except Exception:
            # torch_xcpu keeps the runtime asleep on failure; mirror that so a
            # retry from the router goes through the same path.
            self._state = "SUSPENDED"
            raise
        self._state = "RUNNING"
        logger.info("xcpu device runtime reacquired (tags=%s)", tags)

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import torch_xcpu
        except ImportError:
            return False
        return hasattr(torch_xcpu, "device_runtime_sleep") and hasattr(
            torch_xcpu, "device_runtime_wake_up"
        )

    @classmethod
    def preserves_communicators(cls) -> bool:
        # Communicators are host-side (gloo / MPI) and survive the interval.
        return True

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        # Device memory and pointers stay valid, so compiled code stays valid.
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return True


def register_sleep_mode_backend() -> None:
    """Register the xcpu backend once; safe to call repeatedly."""
    if BACKEND_NAME in SleepModeBackendFactory._registry:
        return
    SleepModeBackendFactory.register_backend(
        BACKEND_NAME, "vllm_xcpu_plugin.sleep_mode", "XcpuSleepModeBackend"
    )
