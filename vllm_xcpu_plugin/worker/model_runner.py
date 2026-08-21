# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def _read_process_memory_bytes() -> dict[str, int]:
    wanted = {"VmRSS", "VmSize", "RssAnon", "RssFile", "RssShmem"}
    result: dict[str, int] = {}
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            fields = value.split()
            if fields:
                result[key.lower() + "_bytes"] = int(fields[0]) * 1024
    return result


class McpuMemoryStatsLogger:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._memory_log_interval_seconds = max(
            0.0,
            float(os.getenv("VLLM_XCPU_MEMORY_LOG_INTERVAL_SECONDS", "0")),
        )
        self._memory_log_interval_steps = max(
            0,
            int(os.getenv("VLLM_XCPU_MEMORY_LOG_INTERVAL_STEPS", "0")),
        )
        self._memory_log_step = 0
        self._memory_log_last_step = 0
        self._memory_log_last_time: float | None = None
        self._memory_log_start_time = time.monotonic()
        if self._memory_log_interval_seconds > 0 or self._memory_log_interval_steps > 0:
            logger.warning(
                "MCPU memory diagnostics enabled: interval_seconds=%s, "
                "interval_steps=%s",
                self._memory_log_interval_seconds,
                self._memory_log_interval_steps,
            )

    def maybe_log(self, scheduler_output) -> None:
        # Refresh at the call site so diagnostics can be adjusted by worker
        # launchers without coupling this helper to their initialization order.
        self._memory_log_interval_seconds = max(
            0.0,
            float(os.getenv("VLLM_XCPU_MEMORY_LOG_INTERVAL_SECONDS", "0")),
        )
        self._memory_log_interval_steps = max(
            0,
            int(os.getenv("VLLM_XCPU_MEMORY_LOG_INTERVAL_STEPS", "0")),
        )
        if (
            self._memory_log_interval_seconds <= 0
            and self._memory_log_interval_steps <= 0
        ):
            return

        num_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        if num_tokens <= 0:
            return

        self._memory_log_step += 1
        now = time.monotonic()
        first_sample = self._memory_log_last_time is None
        seconds_due = (
            self._memory_log_interval_seconds > 0
            and self._memory_log_last_time is not None
            and now - self._memory_log_last_time >= self._memory_log_interval_seconds
        )
        steps_due = (
            self._memory_log_interval_steps > 0
            and self._memory_log_step - self._memory_log_last_step
            >= self._memory_log_interval_steps
        )
        if not (first_sample or seconds_due or steps_due):
            return

        self._memory_log_last_time = now
        self._memory_log_last_step = self._memory_log_step
        try:
            stats = torch.accelerator.memory_stats(self.device)
            keys = (
                "allocated_bytes.all.current",
                "reserved_bytes.all.current",
                "active_bytes.all.current",
                "requested_bytes.all.current",
                "inactive_split_bytes.all.current",
                "allocation.all.current",
                "segment.all.current",
                "active.all.current",
                "inactive_split.all.current",
                "segment.all.allocated",
                "segment.all.freed",
                "num_device_alloc",
                "num_device_free",
                "num_alloc_retries",
                "num_ooms",
                "num_sync_all_streams",
            )
            allocator_stats = {key: int(stats[key]) for key in keys if key in stats}
            payload: dict[str, str | int | float] = {
                "event": "mcpu_memory",
                "pid": os.getpid(),
                "step": self._memory_log_step,
                "elapsed_seconds": round(now - self._memory_log_start_time, 3),
                "num_scheduled_tokens": num_tokens,
                "num_requests": len(
                    getattr(scheduler_output, "num_scheduled_tokens", {})
                ),
                **_read_process_memory_bytes(),
                **allocator_stats,
            }
            reserved = allocator_stats.get("reserved_bytes.all.current")
            active = allocator_stats.get("active_bytes.all.current")
            requested = allocator_stats.get("requested_bytes.all.current")
            if reserved is not None and active is not None:
                payload["cached_bytes"] = reserved - active
            if active is not None and requested is not None:
                payload["active_rounding_overhead_bytes"] = active - requested
            logger.warning("[MCPU_MEMORY] %s", json.dumps(payload, sort_keys=True))
        except Exception:
            logger.exception("Failed to collect MCPU memory diagnostics")


class McpuModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False


class McpuModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    try:
        # Pre-warm TorchDynamo's handler dict BEFORE patching torch.cuda APIs.
        # Patching torch.cuda.current_stream = torch.accelerator.current_stream
        # makes them the same object, which trips an assertion inside
        # TorchInGraphFunctionVariable._get_handlers() (@functools.cache) when
        # it tries to register both as distinct keys in the handlers dict.
        # Calling _get_handlers() here populates the cache with the original
        # distinct function objects so the assertion never fires during compile.
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        TorchInGraphFunctionVariable._get_handlers()

        # replace cuda APIs with mcpu APIs, this should work by default
        torch.cuda.Stream = torch.mcpu.Stream  # type: ignore
        torch.cuda.default_stream = torch.accelerator.current_stream  # type: ignore
        torch.cuda.current_stream = torch.accelerator.current_stream  # type: ignore
        torch.cuda.stream = torch.mcpu.stream  # type: ignore
        torch.cuda.mem_get_info = torch.accelerator.get_memory_info  # type: ignore
        torch.cuda.Event = torch.Event  # type: ignore
        torch.cuda.set_stream = torch.accelerator.set_stream  # type: ignore
        # if supports_xpu_graph():
        #     torch.cuda.graph = torch.xpu.graph
        #     torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        #     torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle
        yield
    finally:
        pass
