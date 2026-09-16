from types import SimpleNamespace

import torch

from vllm_xcpu_plugin.af_ep.attn.client_v7 import ExpertsClientV7


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
