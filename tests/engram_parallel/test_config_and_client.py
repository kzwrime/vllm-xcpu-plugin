# SPDX-License-Identifier: Apache-2.0
"""Tests for EngramParallel config rules, bootstrap validation, session, and client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType
from vllm_xcpu_plugin.engram_disagg.attn.runtime import (
    get_engram_client,
    register_engram_client,
)
from vllm_xcpu_plugin.engram_parallel.bootstrap import (
    bootstrap_engram_parallel_client,
    validate_engram_parallel_config,
)
from vllm_xcpu_plugin.engram_parallel.client import EngramParallelClient
from vllm_xcpu_plugin.engram_parallel.layout import EngramParallelLayout
from vllm_xcpu_plugin.engram_parallel.session import EngramParallelSession
from vllm_xcpu_plugin.models.deepseek_v41.config import update_deepseek_v41_config


class _FakeAttentionConfig:
    def __init__(self, indexer_kv_dtype=None, indexer_sparse_logits=False):
        self.indexer_kv_dtype = indexer_kv_dtype
        self.indexer_sparse_logits = indexer_sparse_logits

    def resolve_indexer_kv_dtype(self, default):
        return default if self.indexer_kv_dtype is None else self.indexer_kv_dtype


def _make_vllm_config(
    *,
    architecture: str = "DeepseekV41ForCausalLM",
    model_type: str = "deepseek_v41",
    load_format: str = "auto",
    pipeline_parallel_size: int = 1,
    enable_dbo: bool = False,
    use_ubatching: bool = False,
    embedding_across_dp: bool = False,
    enable_weights_track: bool = True,
    cpu_offload: bool = True,
    engram_present: bool = True,
    max_num_batched_tokens: int = 64,
):
    hf_text_config = SimpleNamespace(
        engram_layer_ids=[1, 2] if engram_present else [],
        engram_num_embeddings=[97968, 102814],
        engram_max_ngram_size=4,
        engram_vocab_size=4000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=99092,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            hf_config=SimpleNamespace(model_type=model_type),
            hf_text_config=hf_text_config,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            enable_dbo=enable_dbo,
            use_ubatching=use_ubatching,
        ),
        engram_config=(
            SimpleNamespace(
                embedding_across_dp=embedding_across_dp,
                cpu_offload=cpu_offload,
                dp_shared_memory=True,
                use_thp=True,
            )
            if engram_present
            else None
        ),
        load_config=SimpleNamespace(
            load_format=load_format,
            model_loader_extra_config=(
                {"enable_weights_track": True} if enable_weights_track else {}
            ),
            safetensors_load_strategy="lazy",
        ),
        attention_config=_FakeAttentionConfig(),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
    )


def _clear_client():
    import vllm_xcpu_plugin.engram_disagg.attn.runtime as rt

    rt.REGISTRY._client = None


@pytest.fixture(autouse=True)
def _reset_engram_client():
    _clear_client()
    yield
    _clear_client()


def test_update_config_and_validation_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mutual exclusion with ENGRAM_DISAGG.
    monkeypatch.setenv("VLLM_XCPU_ENABLE_ENGRAM_DISAGG", "1")
    monkeypatch.setenv("VLLM_XCPU_ENABLE_ENGRAM_PARALLEL", "1")
    monkeypatch.setenv("VLLM_CPU_USE_MPI", "1")
    with pytest.raises(ValueError, match="cannot both be enabled"):
        update_deepseek_v41_config(_make_vllm_config())
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_engram_parallel_config(_make_vllm_config(), world_size=4)

    monkeypatch.setenv("VLLM_XCPU_ENABLE_ENGRAM_DISAGG", "0")

    # Valid config rewrites load_format, clears offload flags, keeps weights tracking.
    cfg = _make_vllm_config(enable_weights_track=True, cpu_offload=True)
    update_deepseek_v41_config(cfg)
    assert cfg.load_config.load_format == "xcpu_engram_parallel_auto"
    assert cfg.load_config.model_loader_extra_config["enable_weights_track"] is True
    assert cfg.engram_config.cpu_offload is False
    assert cfg.engram_config.dp_shared_memory is False
    assert cfg.engram_config.use_thp is False
    validate_engram_parallel_config(cfg, world_size=4)

    # Dummy load format stays dummy.
    dummy_cfg = _make_vllm_config(load_format="dummy")
    update_deepseek_v41_config(dummy_cfg)
    assert dummy_cfg.load_config.load_format == "dummy"
    validate_engram_parallel_config(dummy_cfg, world_size=4)

    # Rejects non-MPI launcher.
    monkeypatch.setenv("VLLM_CPU_USE_MPI", "0")
    with pytest.raises(ValueError, match="VLLM_CPU_USE_MPI=1"):
        update_deepseek_v41_config(_make_vllm_config())
    monkeypatch.setenv("VLLM_CPU_USE_MPI", "1")

    # Rejects PP>1, DBO, ubatching, embedding_across_dp, wrong model, bad loader knobs.
    with pytest.raises(ValueError, match="embedding_across_dp"):
        update_deepseek_v41_config(_make_vllm_config(embedding_across_dp=True))
    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_engram_parallel_config(_make_vllm_config(pipeline_parallel_size=2), 4)
    with pytest.raises(ValueError, match="DBO"):
        validate_engram_parallel_config(_make_vllm_config(enable_dbo=True), 4)
    with pytest.raises(ValueError, match="DBO"):
        validate_engram_parallel_config(_make_vllm_config(use_ubatching=True), 4)
    with pytest.raises(ValueError, match="DeepseekV41ForCausalLM"):
        validate_engram_parallel_config(_make_vllm_config(architecture="Qwen3"), 4)
    with pytest.raises(ValueError, match="no engram layers"):
        validate_engram_parallel_config(_make_vllm_config(engram_present=False), 4)
    with pytest.raises(ValueError, match="world_size"):
        validate_engram_parallel_config(_make_vllm_config(), 0)

    bad_extra = _make_vllm_config()
    bad_extra.load_config.model_loader_extra_config = {"enable_multithread_load": True}
    with pytest.raises(ValueError, match="enable_multithread_load"):
        validate_engram_parallel_config(bad_extra, 4)

    bad_strat = _make_vllm_config()
    bad_strat.load_config.safetensors_load_strategy = "eager"
    with pytest.raises(ValueError, match="safetensors_load_strategy"):
        validate_engram_parallel_config(bad_strat, 4)

    bad_pcp = _make_vllm_config()
    bad_pcp.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(ValueError, match="prefill context parallelism"):
        validate_engram_parallel_config(bad_pcp, 4)

    bad_dcp = _make_vllm_config()
    bad_dcp.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="decode context parallelism"):
        validate_engram_parallel_config(bad_dcp, 4)

    bad_elastic = _make_vllm_config()
    bad_elastic.parallel_config.enable_elastic_ep = True
    with pytest.raises(ValueError, match="elastic EP"):
        validate_engram_parallel_config(bad_elastic, 4)

    bad_grid = _make_vllm_config()
    bad_grid.parallel_config.data_parallel_size = 2
    bad_grid.parallel_config.data_parallel_rank = 0
    bad_grid.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="!= world_size"):
        validate_engram_parallel_config(bad_grid, 2)


def test_session_dup_free_and_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _FakeDupComm:
        def py2f(self) -> int:
            return 42

        def Free(self) -> None:
            events.append("free")

        def Abort(self, errorcode: int = 1) -> None:
            events.append(f"abort:{errorcode}")

    class _FakeClusterComm:
        def Dup(self):
            events.append("dup")
            return _FakeDupComm()

    fake_world = SimpleNamespace(
        cluster_type=ClusterType.ATTN,
        cluster_size=4,
        cluster_rank=1,
        cluster_comm=_FakeClusterComm(),
    )

    with pytest.raises(ValueError, match="ATTN worker ranks only"):
        EngramParallelSession(
            SimpleNamespace(cluster_type=ClusterType.MOE),
            world_rank=0,
            world_size=1,
            max_tokens_proposal=8,
        )
    with pytest.raises(ValueError, match="rank mismatch"):
        EngramParallelSession(
            fake_world, world_rank=0, world_size=4, max_tokens_proposal=8
        )
    with pytest.raises(ValueError, match="positive max_tokens_proposal"):
        EngramParallelSession(
            fake_world, world_rank=1, world_size=4, max_tokens_proposal=0
        )

    monkeypatch.setenv("VLLM_XCPU_ENABLE_ENGRAM_DISAGG", "0")
    client = bootstrap_engram_parallel_client(
        mpi_world=fake_world,
        vllm_config=_make_vllm_config(max_num_batched_tokens=32),
        world_rank=1,
        world_size=4,
    )
    assert get_engram_client() is client
    assert events == ["dup"]

    with pytest.raises(RuntimeError, match="already registered"):
        bootstrap_engram_parallel_client(
            mpi_world=fake_world,
            vllm_config=_make_vllm_config(),
            world_rank=1,
            world_size=4,
        )

    # Stopping before client.initialize() aborts the communicator rather than
    # calling collective MPI_Comm_free() while peers may be in MPI_Allgather.
    client.stop()
    assert events == ["dup", "abort:1"]
    # stop() is idempotent.
    client.stop()
    assert events == ["dup", "abort:1"]

    # Direct EngramParallelSession.close() frees cleanly.
    session2 = EngramParallelSession(
        fake_world, world_rank=1, world_size=4, max_tokens_proposal=8
    )
    session2.close()
    assert events == ["dup", "abort:1", "dup", "free"]

    # Real registry works cleanly with register_engram_client when cleared.
    _clear_client()
    register_engram_client(client)
    assert get_engram_client() is client


def test_client_sp_window_initialize_and_state_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple]] = []

    layout = EngramParallelLayout.from_sizes(
        layer_ids=(1, 14),
        num_embeddings=(100, 120),
        head_sizes=((20, 30, 40), (30, 35, 45)),
        world_size=4,
    )

    class _StubSession:
        world_rank = 1
        world_size = 4
        max_tokens_proposal = 10
        communicator_handle = torch.tensor([7], dtype=torch.int64)

        def initialize(self, lay, weights, scales, *, max_tokens_proposal=None, **kw):
            del lay
            calls.append(
                (
                    "session.initialize",
                    (
                        max_tokens_proposal,
                        len(weights),
                        len(scales),
                        kw.get("dp_size"),
                        kw.get("dp_rank"),
                        kw.get("tp_size"),
                        kw.get("tp_rank"),
                        kw.get("use_sequence_parallel"),
                    ),
                )
            )
            return max_tokens_proposal

        def require_initialized(self):
            return layout, 3

        def close(self):
            calls.append(("session.close", ()))

        def abort(self, errorcode: int = 1):
            calls.append(("session.abort", (errorcode,)))

    def _fake_request(h, s, c, _rs, _rr, _ri, _cm, dt=None):
        calls.append(
            (
                "request",
                (
                    tuple(h.shape),
                    s,
                    c,
                    tuple(dt.tolist()) if dt is not None else None,
                ),
            )
        )

    fake_ops = SimpleNamespace(
        deepseek_v41_engram_parallel_slot_ints=lambda mt, mc: 1 + mt * mc,
        deepseek_v41_engram_parallel_slot_bytes=lambda mt, mc: mt * mc * 264,
        deepseek_v41_engram_parallel_request=_fake_request,
        deepseek_v41_engram_parallel_collect=lambda *_: calls.append(("collect", ())),
        deepseek_v41_engram_parallel_decode=lambda rows, pr, ri, idx, nt, ly, mq: (
            calls.append(("decode", (idx, nt, ly, mq, tuple(rows.shape))))
        ),
        deepseek_v41_engram_parallel_cleanup=lambda: calls.append(("cleanup", ())),
    )
    monkeypatch.setattr("torch_xcpu.ops", fake_ops, raising=False)
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)

    client = EngramParallelClient(_StubSession(), layout)

    # prepare_embedding + consume_embedding_slice check dimensions & propagate slice.
    fake_engram_layout = SimpleNamespace(
        layer_ids=[1, 14],
        num_embeddings=[100, 120],
        primes=[[[20, 30, 40]], [[30, 35, 45]]],
    )
    with pytest.raises(RuntimeError, match="prepare_embedding"):
        client.consume_embedding_slice(100, (20, 30, 40))

    lo0, hi0 = client.prepare_embedding(fake_engram_layout, 0)
    assert (lo0, hi0) == client.consume_embedding_slice(100, (20, 30, 40))

    with pytest.raises(RuntimeError, match="dimensions mismatch"):
        client.prepare_embedding(fake_engram_layout, 1)
        client.consume_embedding_slice(999, (30, 35, 45))

    # Must bind all layers before initialize; weights and scales required.
    with pytest.raises(RuntimeError, match="local table weight and scale"):
        client.bind_layer(True, 0, weight=None, scale=torch.empty(0))

    w0 = torch.empty(layout.layer_slice(0, 1)[1] - layout.layer_slice(0, 1)[0], 256)
    s0 = torch.empty(w0.shape[0], 8)
    w1 = torch.empty(layout.layer_slice(1, 1)[1] - layout.layer_slice(1, 1)[0], 256)
    s1 = torch.empty(w1.shape[0], 8)

    client.bind_layer(True, 0, weight=w0, scale=s0)
    with pytest.raises(RuntimeError, match="do not cover"):
        client.initialize()
    with pytest.raises(RuntimeError, match="disagree on sequence parallelism"):
        client.bind_layer(False, 1, weight=w1, scale=s1)
    client.bind_layer(True, 1, weight=w1, scale=s1)

    # SP window scaling on TP=4, rank=1 and capacity negotiation scaling ceil(10/4)=3.
    client._tp = (4, 1)
    assert client.local_window(10) == (3, 3)
    client._tp = (4, 3)
    assert client.local_window(10) == (9, 1)
    client._tp = (4, 1)

    client.initialize_from_engram_layout(fake_engram_layout)
    assert ("session.initialize", (3, 2, 2, 1, 0, 4, 1, True)) in calls

    # Errors before request / ordering / buffer sizes / compile guard.
    hashes = torch.zeros(10, 2, 3, dtype=torch.int32)
    rows_small = torch.empty(1, 3, 256, dtype=torch.bfloat16)
    rows_ok = torch.empty(3, 3, 256, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="before any request"):
        client.fill_rows(0, rows_ok)

    client.request(hashes)
    assert ("request", ((10, 2, 3), 3, 3, (10,))) in calls
    with pytest.raises(RuntimeError, match="was never collected"):
        client.request(hashes)
    with pytest.raises(RuntimeError, match="layer 0 must collect"):
        client.fill_rows(1, rows_ok)
    with pytest.raises(RuntimeError, match="holds 1 tokens"):
        client.fill_rows(0, rows_small)

    client.fill_rows(0, rows_ok)
    with pytest.raises(RuntimeError, match="already collected"):
        client.fill_rows(0, rows_ok)
    client.fill_rows(1, rows_ok)

    # Compile-graph guard on request and collect.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="outside compiled graphs"):
        client.request(hashes)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    client.stop()
    assert ("cleanup", ()) in calls
    assert ("session.close", ()) in calls
    assert all(op != "session.abort" for op, _ in calls)


def test_client_stop_aborts_uncollected_without_cleanup_or_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, tuple]] = []
    layout = EngramParallelLayout.from_sizes(
        layer_ids=(1,),
        num_embeddings=(40,),
        head_sizes=((10, 10, 10),),
        world_size=2,
    )

    class _StubSession:
        world_rank = 0
        world_size = 2
        max_tokens_proposal = 4
        communicator_handle = torch.tensor([9], dtype=torch.int64)

        def initialize(self, lay, weights, scales, *, max_tokens_proposal=None, **kw):
            del lay, weights, scales, kw
            return max_tokens_proposal

        def require_initialized(self):
            return layout, 4

        def close(self):
            events.append(("close", ()))

        def abort(self, errorcode: int = 1):
            events.append(("abort", (errorcode,)))

    fake_ops = SimpleNamespace(
        deepseek_v41_engram_parallel_slot_ints=lambda mt, mc: 1 + mt * mc,
        deepseek_v41_engram_parallel_slot_bytes=lambda mt, mc: mt * mc * 264,
        deepseek_v41_engram_parallel_request=lambda *_: events.append(("request", ())),
        deepseek_v41_engram_parallel_cleanup=lambda: events.append(("cleanup", ())),
    )
    monkeypatch.setattr("torch_xcpu.ops", fake_ops, raising=False)
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)

    client = EngramParallelClient(_StubSession(), layout)
    w0 = torch.empty(layout.layer_slice(0, 0)[1] - layout.layer_slice(0, 0)[0], 256)
    s0 = torch.empty(w0.shape[0], 8)
    client.bind_layer(False, 0, weight=w0, scale=s0)
    client.initialize(layout)
    client.request(torch.zeros(2, 1, 3, dtype=torch.int32))

    # An uncollected in-flight request must abort without calling cleanup(),
    # MPI_Comm_free(), or releasing host MPI buffers.
    client.stop()
    assert events == [("request", ()), ("abort", (1,))]
    assert client._req_send is not None
    assert client._req_recv is not None
    assert client._reply_send is not None
    assert client._reply_recv is not None
    # Stopping again remains idempotent.
    client.stop()
    assert events == [("request", ()), ("abort", (1,))]


def _run_dummy_hashes_case(
    monkeypatch: pytest.MonkeyPatch,
    do_not_compile: bool,
    input_ids: torch.Tensor | None,
) -> None:
    from vllm_xcpu_plugin.models.deepseek_v41 import model as v41_model

    seen: dict[str, object] = {}
    prepared: list[tuple[int, tuple[int, ...]]] = []

    class _FakeHashState:
        def dummy_hashes(self, ref_tensor: torch.Tensor):
            seen["dummy_input_shape"] = tuple(ref_tensor.shape)
            n = ref_tensor.shape[0]
            return (
                torch.full((n, 2, 24), -1, dtype=torch.int32),
                torch.zeros(n, dtype=torch.bool),
            )

    class _FakeClient:
        def request(self, hashes: torch.Tensor) -> None:
            seen["request_shape"] = tuple(hashes.shape)

    class _FakeEngram:
        def __init__(self, idx: int) -> None:
            self.layer_hash_index = idx

        def prepare_embeddings(self, layer_hashes: torch.Tensor) -> None:
            prepared.append((self.layer_hash_index, tuple(layer_hashes.shape)))

    class _FakeSourceLayers:
        def __init__(self, flag: bool) -> None:
            self.do_not_compile = flag

        def __call__(
            self,
            hidden_states,
            positions,
            inp_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            engram_hashes,
            engram_mask,
        ):
            seen["source_hashes_shape"] = tuple(engram_hashes.shape)
            seen["source_mask_shape"] = tuple(engram_mask.shape)
            return hidden_states, residual, post_mix, res_mix, pre_mix

    fake_client = _FakeClient()
    monkeypatch.setattr(v41_model, "get_engram_client", lambda: fake_client)

    m = v41_model.XcpuDeepseekV41Model.__new__(v41_model.XcpuDeepseekV41Model)
    m.source_layer_ids = range(0, 2)
    m.start_layer = 0
    m.end_layer = 2
    m.engram_dp_shared_memory = False
    m.engram_hash = _FakeHashState()
    m.source_layers = _FakeSourceLayers(do_not_compile)
    m.source_aux_layer_ids = lambda: []
    m.layers = [
        SimpleNamespace(engram=None),
        SimpleNamespace(engram=_FakeEngram(0)),
    ]

    positions = torch.zeros(16, dtype=torch.int64)
    hidden = torch.zeros(8, 4, 16)
    m._run_layers(
        m.source_layer_ids,
        hidden,
        positions,
        input_ids,
        None,
        None,
        None,
        None,
        {},
        engram_hashes=None,
        engram_mask=None,
    )
    assert seen["dummy_input_shape"] == (16,)
    assert seen["request_shape"] == (16, 2, 24)
    assert prepared == [(0, (16, 24))]
    assert seen["source_hashes_shape"] == (16, 2, 24)
    assert seen["source_mask_shape"] == (16,)


def test_client_dp_metadata_and_coordination_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.v1.worker.dp_utils import skip_dp_coordination

    captured: list[tuple[int, int, list[int]]] = []
    layout = EngramParallelLayout.from_sizes(
        layer_ids=(1,),
        num_embeddings=(40,),
        head_sizes=((10, 10, 10, 10),),
        world_size=4,
    )

    class _StubSession:
        world_rank = 2
        world_size = 4
        max_tokens_proposal = 8
        communicator_handle = torch.tensor([11], dtype=torch.int64)

        def initialize(self, lay, weights, scales, *, max_tokens_proposal=None, **kw):
            del lay, weights, scales, kw
            return max_tokens_proposal

        def require_initialized(self):
            return layout, 8

        def close(self):
            return None

        def abort(self, errorcode: int = 1):
            del errorcode
            return None

    fake_ops = SimpleNamespace(
        deepseek_v41_engram_parallel_slot_ints=lambda mt, mc: 1 + mt * mc,
        deepseek_v41_engram_parallel_slot_bytes=lambda mt, mc: mt * mc * 264,
        deepseek_v41_engram_parallel_request=lambda h, s, c, _rs, _rr, _ri, _cm, dt: (
            captured.append((s, c, dt.tolist()))
        ),
        deepseek_v41_engram_parallel_collect=lambda *_: None,
        deepseek_v41_engram_parallel_decode=lambda *_: None,
        deepseek_v41_engram_parallel_cleanup=lambda: None,
    )
    monkeypatch.setattr("torch_xcpu.ops", fake_ops, raising=False)
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)

    with pytest.raises(ValueError, match="dp_size and dp_rank"):
        EngramParallelClient(_StubSession(), layout, dp_size=2)
    with pytest.raises(ValueError, match="invalid DP topology"):
        EngramParallelClient(_StubSession(), layout, dp_size=2, dp_rank=2)

    client = EngramParallelClient(_StubSession(), layout, dp_size=2, dp_rank=1)
    client._tp = (2, 0)
    w0 = torch.empty(layout.layer_slice(0, 2)[1] - layout.layer_slice(0, 2)[0], 256)
    s0 = torch.empty(w0.shape[0], 8)
    client.bind_layer(True, 0, weight=w0, scale=s0)
    client.initialize(layout)

    hashes = torch.zeros(3, 1, 4, dtype=torch.int32)
    with override_forward_context(None), pytest.raises(
        RuntimeError, match="active ForwardContext"
    ):
        client.request(hashes)

    empty_ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    with override_forward_context(empty_ctx), pytest.raises(
        RuntimeError, match="ForwardContext.dp_metadata"
    ):
        client.request(hashes)

    good_dp = torch.tensor([5, 3], dtype=torch.int32)
    good_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=good_dp),
    )
    with override_forward_context(good_ctx), skip_dp_coordination(), pytest.raises(
        RuntimeError, match="skip_dp_coordination"
    ):
        client.request(hashes)

    bad_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor([5, 4], dtype=torch.int32)
        ),
    )
    with override_forward_context(bad_ctx), pytest.raises(
        RuntimeError, match="disagree with dp_metadata"
    ):
        client.request(hashes)

    with override_forward_context(good_ctx):
        client.request(hashes)
    assert captured == [(0, 2, [5, 3])]
    client.fill_rows(0, torch.empty(2, 4, 256, dtype=torch.bfloat16))
    client.stop()


def test_model_run_layers_dummy_hashes_uses_pre_sp_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When forward enters _run_layers with engram_hashes=None (e.g. inputs_embeds
    or capture warmup), dummy_hashes must size off pre-SP `positions` rather than
    SP-sharded `input_ids` and still dispatch client.request even in eager mode."""
    # Compile warmup with SP-sharded input_ids, then eager inputs_embeds (None).
    for do_not_compile, input_ids in [
        (False, torch.zeros(8, dtype=torch.int32)),
        (True, None),
    ]:
        _run_dummy_hashes_case(monkeypatch, do_not_compile, input_ids)
