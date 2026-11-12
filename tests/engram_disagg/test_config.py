"""The Engram disaggregation config update: storage knobs are forced off with
a log line, conflicting lookup semantics fail loudly."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.models.deepseek_v41.config import _update_engram_disagg_config

LOGGER_NAME = "vllm_xcpu_plugin.models.deepseek_v41.config"


def make_config(**engram_overrides) -> SimpleNamespace:
    engram = SimpleNamespace(
        embedding_across_dp=False,
        cpu_offload=False,
        dp_shared_memory=False,
        use_thp=False,
    )
    for name, value in engram_overrides.items():
        setattr(engram, name, value)
    return SimpleNamespace(
        load_config=SimpleNamespace(load_format="dummy"),
        engram_config=engram,
    )


def update(config: SimpleNamespace, model_type: str = "deepseek_v41_flash") -> None:
    _update_engram_disagg_config(config, config.engram_config, model_type)


@pytest.fixture(autouse=True)
def mpi_launcher(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_USE_MPI", "1")


def test_storage_knobs_are_forced_off_and_logged(caplog):
    config = make_config(cpu_offload=True, use_thp=True)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        update(config)
    assert not (
        config.engram_config.cpu_offload
        or config.engram_config.dp_shared_memory
        or config.engram_config.use_thp
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ignoring engram_config cpu_offload, use_thp" in message for message in messages
    )


def test_no_override_log_when_knobs_already_off(caplog):
    config = make_config()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        update(config)
    assert not any(
        "ignoring engram_config" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize(
    "model_type", ["deepseek_v41", "deepseek_v41_flash", "deepseek_v41_vl"]
)
def test_accepts_v41_model_types(model_type):
    config = make_config()
    update(config, model_type)


def test_rejects_wrong_model_missing_engram_and_no_mpi(monkeypatch):
    with pytest.raises(ValueError, match="needs a DeepSeek-V4.1 model"):
        update(make_config(), model_type="qwen3_next")
    with pytest.raises(ValueError, match="needs a DeepSeek-V4.1 model"):
        _update_engram_disagg_config(
            SimpleNamespace(load_config=SimpleNamespace(load_format="dummy")),
            None,
            "deepseek_v41",
        )
    monkeypatch.setenv("VLLM_CPU_USE_MPI", "0")
    with pytest.raises(ValueError, match="requires the MPI launcher"):
        update(make_config())


def test_embedding_across_dp_is_rejected_not_silently_overridden():
    config = make_config(embedding_across_dp=True)
    with pytest.raises(ValueError, match="embedding_across_dp"):
        update(config)
    assert config.engram_config.embedding_across_dp


def test_env_flag_is_read_lazily():
    # The launcher sets the env before the engine boots; nothing caches it.
    assert envs_xcpu.VLLM_CPU_USE_MPI is True
