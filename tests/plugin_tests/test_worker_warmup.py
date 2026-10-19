# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from vllm.config import CompilationMode

from vllm_xcpu_plugin.worker import worker_v1


@pytest.mark.parametrize(
    "v2,mode,backend,options,expected",
    [
        (True, CompilationMode.DYNAMO_TRACE_ONCE, "inductor", {}, True),
        (False, CompilationMode.DYNAMO_TRACE_ONCE, "inductor", {}, False),
        (True, CompilationMode.NONE, "inductor", {}, False),
        (True, CompilationMode.VLLM_COMPILE, "inductor", {}, False),
        (True, CompilationMode.DYNAMO_TRACE_ONCE, "eager", {}, False),
        (
            True,
            CompilationMode.DYNAMO_TRACE_ONCE,
            "inductor",
            {"memory_planning": False},
            False,
        ),
        (
            True,
            CompilationMode.DYNAMO_TRACE_ONCE,
            "inductor",
            {"memory_pool": "none"},
            False,
        ),
    ],
)
def test_pool_warmup_order(monkeypatch, v2, mode, backend, options, expected):
    worker = object.__new__(worker_v1.McpuWorker)
    worker.use_v2_model_runner = v2
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=mode, backend=backend, inductor_compile_config=options
        )
    )
    worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024)
    worker.model_config = SimpleNamespace(seed=42)
    worker.model_runner = SimpleNamespace(get_model=torch.nn.Module)
    events = Mock()
    monkeypatch.setattr(worker_v1.Worker, "compile_or_warm_up_model", events.compile)
    monkeypatch.setattr(torch.accelerator, "synchronize", events.sync)
    monkeypatch.setattr(torch.accelerator, "empty_cache", events.trim)
    monkeypatch.setattr(worker_v1, "set_random_seed", events.seed)
    worker.model_runner._dummy_run = events.run
    with torch._inductor.config.patch(
        memory_planning=True, memory_pool="intermediates"
    ):
        result = worker.compile_or_warm_up_model()
    assert result is events.compile.return_value
    expected_calls = [call.compile()]
    if expected:
        expected_calls += [
            call.sync(),
            call.trim(),
            call.run(1024, skip_eplb=True),
            call.sync(),
            call.seed(42),
        ]
    assert events.mock_calls == expected_calls
