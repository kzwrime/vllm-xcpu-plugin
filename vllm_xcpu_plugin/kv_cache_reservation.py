"""Reserve the KV cache's device memory before the weights are loaded.

vLLM allocates the whole KV cache as one tensor, and only after model loading
and the profiling run. By then the device heap holds the weights with holes
left by freed temporaries, so that single multi-GiB request can fail although
enough memory is free in total. The reservation takes a VLLM_CPU_KVCACHE_SPACE
sized tensor in init_device while the heap is still empty and keeps it alive
through loading and profiling. The model runner frees it right before vLLM
allocates the KV cache: the caching allocator then serves the KV tensor from
that cached block (the KV size never exceeds the reservation), so the device
heap is not asked for anything.

Nothing else may allocate between the release and the KV allocation, or the
best-fit search would carve the reservation up for it. The release therefore
sits inside the runner's allocation hook, not around initialize_kv_cache.
"""

import torch
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

logger = init_logger(__name__)
_reservation: torch.Tensor | None = None


def reserve(nbytes: int, device: torch.device) -> None:
    global _reservation
    assert _reservation is None, "KV cache already reserved"
    _reservation = torch.empty(nbytes, dtype=torch.int8, device=device)
    logger.info(
        "Reserved %s GiB of device memory for the KV cache at %#x",
        format_gib(nbytes),
        _reservation.data_ptr(),
    )


def reserved_bytes() -> int:
    return 0 if _reservation is None else _reservation.numel()


def handoff(allocate, *args, **kwargs):
    """Free the reservation, run the KV cache allocation, check the reuse."""
    global _reservation
    reservation, _reservation = _reservation, None
    if reservation is None:
        return allocate(*args, **kwargs)
    ptr, nbytes = reservation.data_ptr(), reservation.numel()
    device = reservation.device
    del reservation
    kv_caches = allocate(*args, **kwargs)
    backing = {
        t.untyped_storage().data_ptr()
        for t in kv_caches.values()
        if isinstance(t, torch.Tensor) and t.device == device
    }
    if backing == {ptr}:
        logger.info("KV cache placed in the %s GiB reservation", format_gib(nbytes))
    elif backing:
        # The allocation still succeeded; the reservation stays cached and the
        # allocator returns it to the device on its next out-of-memory retry.
        logger.warning(
            "KV cache backing %s does not reuse the %s GiB reservation at %#x",
            [f"{p:#x}" for p in sorted(backing)],
            format_gib(nbytes),
            ptr,
        )
    return kv_caches
