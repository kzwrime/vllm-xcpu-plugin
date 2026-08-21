import json
from types import SimpleNamespace

import torch

from vllm_xcpu_plugin.worker import model_runner


def test_memory_stats_logging_reports_allocator_and_process_memory(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_XCPU_MEMORY_LOG_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("VLLM_XCPU_MEMORY_LOG_INTERVAL_STEPS", "2")
    monkeypatch.setattr(
        torch.accelerator,
        "memory_stats",
        lambda device: {
            "allocated_bytes.all.current": 60,
            "reserved_bytes.all.current": 100,
            "active_bytes.all.current": 70,
            "requested_bytes.all.current": 55,
            "inactive_split_bytes.all.current": 20,
            "segment.all.current": 5,
            "num_ooms": 0,
        },
    )
    messages = []
    monkeypatch.setattr(
        model_runner.logger,
        "warning",
        lambda message, *args: messages.append((message, args)),
    )

    monitor = model_runner.McpuMemoryStatsLogger(torch.device("cpu"))
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"request": 1},
    )

    monitor.maybe_log(scheduler_output)
    monitor.maybe_log(scheduler_output)
    monitor.maybe_log(scheduler_output)

    samples = [item for item in messages if item[0] == "[MCPU_MEMORY] %s"]
    assert len(samples) == 2
    payload = json.loads(samples[0][1][0])
    assert payload["cached_bytes"] == 30
    assert payload["active_rounding_overhead_bytes"] == 15
    assert payload["inactive_split_bytes.all.current"] == 20
    assert payload["num_ooms"] == 0
    assert payload["num_scheduled_tokens"] == 1
    assert payload["num_requests"] == 1
    assert payload["vmrss_bytes"] > 0
