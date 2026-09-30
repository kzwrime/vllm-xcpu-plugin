from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_xcpu_plugin.af_ep.attn.client_v7 import ExpertsClientV7
from vllm_xcpu_plugin.af_ep.attn.compatibility import (
    AttentionSupport,
    support_from_vllm,
    validate_attention_support,
)


def test_attention_initializes_transport_after_model_load(monkeypatch, make_af_session):
    import torch_xcpu

    from vllm_xcpu_plugin.worker.worker_v1 import McpuWorker, Worker

    calls = []
    monkeypatch.setattr(Worker, "load_model", lambda self, **kw: calls.append("load"))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: calls.append("drain"))
    monkeypatch.setattr(
        torch_xcpu.ops,
        "moe_af_v7_initialize",
        lambda *args: calls.append(("initialize", *args[:4])),
    )
    worker = object.__new__(McpuWorker)
    worker.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(hidden_size=16, num_experts_per_tok=6),
        dtype=torch.bfloat16,
    )
    worker._af_client = ExpertsClientV7(make_af_session())

    worker.load_model()

    assert calls == ["load", "drain", ("initialize", 8, 16, 6, torch.bfloat16)]


def test_attention_and_expert_rank_counts_are_independent():
    support = AttentionSupport(
        use_v2_model_runner=True,
        enable_expert_parallel=True,
        logical_ep_size=4,
        dtype="torch.bfloat16",
    )

    validate_attention_support(1, support)
    validate_attention_support(4, support)
    validate_attention_support(4, replace(support, logical_ep_size=1))


def test_attention_accepts_compile_and_rejects_graph_capture():
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=2,
            enable_dbo=False,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_experts=8),
            enforce_eager=False,
            dtype=torch.bfloat16,
            quantization=None,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        lora_config=None,
        speculative_config=None,
    )
    validate_attention_support(2, support_from_vllm(config))

    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(ValueError, match="CUDAGraph capture is not supported"):
        validate_attention_support(2, support_from_vllm(config))
