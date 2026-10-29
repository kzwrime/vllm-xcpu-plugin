"""XcpuSleepModeBackend: registration, state tracking and idempotent resume."""

import pytest
import torch_xcpu
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

from vllm_xcpu_plugin.platform import McpuPlatform
from vllm_xcpu_plugin.sleep_mode import (
    BACKEND_NAME,
    XcpuSleepModeBackend,
    register_sleep_mode_backend,
)


@pytest.fixture(autouse=True)
def awake_runtime():
    torch_xcpu.initialize_runtime()
    if torch_xcpu.is_device_runtime_sleeping():
        torch_xcpu.device_runtime_wake_up()
    yield
    if torch_xcpu.is_device_runtime_sleeping():
        torch_xcpu.device_runtime_wake_up()


def test_registration_is_idempotent():
    register_sleep_mode_backend()
    register_sleep_mode_backend()
    assert SleepModeBackendFactory.get_backend_class(BACKEND_NAME) is (
        XcpuSleepModeBackend
    )
    assert XcpuSleepModeBackend.is_supported()
    assert McpuPlatform().is_sleep_mode_available()
    assert XcpuSleepModeBackend.preserves_communicators()
    assert XcpuSleepModeBackend.preserves_compiled_artifacts()


def test_suspend_resume_tracks_runtime_state():
    backend = XcpuSleepModeBackend()
    assert backend.state() == "RUNNING"

    backend.suspend(1)
    assert backend.state() == "SUSPENDED"
    assert torch_xcpu.is_device_runtime_sleeping()
    # Executor-level double sleep is tolerated.
    backend.suspend(1)
    assert torch_xcpu.is_device_runtime_sleeping()

    # Tagged partial wake-ups reacquire on the first call and no-op after.
    backend.resume(["weights"])
    assert backend.state() == "RUNNING"
    assert not torch_xcpu.is_device_runtime_sleeping()
    backend.resume(["kv_cache"])
    assert backend.state() == "RUNNING"

    with pytest.raises(RuntimeError):
        backend.discard(("kv_cache",))


def test_enable_sleep_mode_selects_xcpu_backend():
    from vllm.config import ModelConfig, VllmConfig

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B", enable_sleep_mode=True, skip_tokenizer_init=True
    )
    assert model_config.sleep_mode_backend == "cumem"
    config = VllmConfig(model_config=model_config)
    assert config.model_config.sleep_mode_backend == BACKEND_NAME
