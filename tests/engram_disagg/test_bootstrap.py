"""Unsupported configurations fail before model load; supported ones register
a client."""

from types import SimpleNamespace

import pytest

from vllm_xcpu_plugin.engram_disagg.attn import runtime
from vllm_xcpu_plugin.engram_disagg.attn.bootstrap import (
    bootstrap_engram_client,
    validate_attention_config,
)


def make_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(
                engram_layer_ids=[1, 14], engram_n_heads=8, engram_max_ngram_size=4
            ),
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, enable_dbo=False, use_ubatching=False
        ),
        engram_config=SimpleNamespace(embedding_across_dp=False),
        load_config=SimpleNamespace(
            load_format="xcpu_engram_auto",
            model_loader_extra_config={},
            safetensors_load_strategy=None,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
    )
    for key, value in overrides.items():
        target, _, attr = key.rpartition(".")
        setattr(eval(f"config.{target}") if target else config, attr, value)
    return config


def test_validate_accepts_the_supported_shape(monkeypatch):
    monkeypatch.setenv("VLLM_XCPU_ENABLE_AF_EP", "0")
    for ranks in (1, 2, 3, 4, 6, 8, 12, 24):
        validate_attention_config(make_config(), ranks)


@pytest.mark.parametrize(
    "overrides,ranks,message",
    [
        ({}, 5, "cannot be split"),
        ({"parallel_config.pipeline_parallel_size": 2}, 2, "pipeline parallelism"),
        ({"parallel_config.enable_dbo": True}, 2, "DBO"),
        ({"engram_config.embedding_across_dp": True}, 2, "embedding_across_dp"),
        ({"load_config.load_format": "xcpu_engram_npcache"}, 2, "load format"),
        (
            {"load_config.model_loader_extra_config": {"enable_weights_track": True}},
            2,
            "enable_weights_track",
        ),
        (
            {"load_config.safetensors_load_strategy": "eager"},
            2,
            "safetensors_load_strategy",
        ),
        (
            {"model_config.architecture": "Qwen4ExpForCausalLM"},
            2,
            "DeepseekV41ForCausalLM",
        ),
    ],
)
def test_validate_rejects_unsupported(monkeypatch, overrides, ranks, message):
    monkeypatch.setenv("VLLM_XCPU_ENABLE_AF_EP", "0")
    with pytest.raises(ValueError, match=message):
        validate_attention_config(make_config(**overrides), ranks)


def test_bootstrap_registers_one_client(monkeypatch, world):
    monkeypatch.setenv("VLLM_XCPU_ENABLE_AF_EP", "0")
    monkeypatch.setattr(runtime.REGISTRY, "_client", None)
    client = bootstrap_engram_client(mpi_world=world(), vllm_config=make_config())
    assert runtime.get_engram_client() is client
    assert client.session.max_tokens_proposal == 32
    with pytest.raises(RuntimeError, match="already registered"):
        bootstrap_engram_client(mpi_world=world(), vllm_config=make_config())
    runtime.unregister_engram_client(client)
    assert runtime.get_engram_client() is None
