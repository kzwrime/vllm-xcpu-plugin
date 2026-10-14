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


@pytest.mark.parametrize("version", [7, 8])
def test_bootstrap_selects_requested_transport(monkeypatch, make_af_session, version):
    from vllm_xcpu_plugin.af_ep.attn import bootstrap
    from vllm_xcpu_plugin.af_ep.attn.client_v8 import ExpertsClientV8
    from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

    # Reuse session construction while testing the real backend selection.
    session = make_af_session(version=version)
    monkeypatch.setattr(bootstrap, f"AfV{version}Session", lambda *a, **kw: session)
    monkeypatch.setattr(bootstrap, "get_remote_experts_client", lambda: None)
    registered = []
    monkeypatch.setattr(bootstrap, "register_remote_experts_client", registered.append)
    monkeypatch.setattr(bootstrap, "support_from_vllm", lambda config: None)
    monkeypatch.setattr(bootstrap, "validate_attention_support", lambda *args: None)
    client = bootstrap.bootstrap_attention_worker(
        mpi_world=SimpleNamespace(
            cluster_type=ClusterType.ATTN, cluster_size=2, cluster_rank=0
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                all2all_backend=f"mpi_alltoallv_v{version}"
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        ),
        model_world_rank=0,
        model_world_size=2,
    )
    assert isinstance(client, ExpertsClientV8 if version == 8 else ExpertsClientV7)
    assert registered == [client]
    assert client._session.metadata[0].item() == (2 if version == 8 else 1)


@pytest.mark.parametrize("version", [7, 8])
@pytest.mark.parametrize(
    "sp_sizes,expected", [([1], 33), ([2], 17), ([4], 9), ([2, 1], 33), ([4, 2], 17)]
)
def test_af_capacity_is_negotiated_after_sp_once(
    make_af_session, version, sp_sizes, expected
):
    from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

    attention = make_af_session(
        max_rows=33, num_attention_ranks=4, num_expert_ranks=2, version=version
    )
    expert = make_af_session(
        ClusterType.MOE,
        max_rows=33,
        num_attention_ranks=4,
        num_expert_ranks=2,
        version=version,
    )
    for size in sp_sizes:
        attention.register_layer_capacity(size)
    proposals = [expected] * 4 + [0] * 2
    seen = []
    for session in (attention, expert):
        session._global_world_comm.allgather = lambda n: seen.append(n) or proposals
        session.initialize(16, 6, torch.bfloat16)
        session.initialize(16, 6, torch.bfloat16)
        assert session.max_rows_per_attention_rank == expected
        assert session.expert_capacity == 4 * expected
    assert seen == [
        expected,
        0,
    ]  # F never divides by SP again, initialize is idempotent.


@pytest.mark.parametrize("version", [7, 8])
def test_af_capacity_uses_largest_attention_proposal(make_af_session, version):
    session = make_af_session(max_rows=33, version=version)
    session.register_layer_capacity(4)
    session._global_world_comm.allgather = lambda _: [9, 17, 0, 0]
    session.initialize(16, 6, torch.bfloat16)
    assert session.max_rows_per_attention_rank == 17
