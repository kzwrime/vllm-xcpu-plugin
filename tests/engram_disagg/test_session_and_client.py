"""Engram disaggregation world checks, shard layout and the client state machine."""

from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiCluster
from vllm_xcpu_plugin.engram_disagg.attn.client import RemoteEngramClient
from vllm_xcpu_plugin.engram_disagg.common import EngramSession, EngramShardLayout

HEAD_SIZES = tuple(tuple(10 + c for c in range(24)) for _ in range(2))


def make_shard(num_engram_ranks=2):
    return EngramShardLayout(
        layer_ids=(1, 14),
        num_embeddings=(sum(HEAD_SIZES[0]), sum(HEAD_SIZES[1])),
        head_sizes=HEAD_SIZES,
        num_engram_ranks=num_engram_ranks,
    )


def test_shard_layout_cuts_contiguous_columns():
    shard = make_shard(num_engram_ranks=3)
    assert shard.cols_per_rank == 8
    assert shard.column_range(0, 0) == (0, sum(HEAD_SIZES[0][:8]))
    assert shard.column_range(1, 2) == (sum(HEAD_SIZES[1][:16]), sum(HEAD_SIZES[1]))
    with pytest.raises(ValueError, match="cannot be split evenly"):
        make_shard(num_engram_ranks=5)


def test_shard_layout_from_vllm_engram_layout():
    class Layout:
        layer_ids = (1, 14)
        num_embeddings = (sum(HEAD_SIZES[0]), sum(HEAD_SIZES[1]))
        primes = tuple(
            tuple(tuple(sizes[n * 8 : (n + 1) * 8]) for n in range(3))
            for sizes in HEAD_SIZES
        )

    shard = EngramShardLayout.from_engram_layout(Layout(), 8)
    assert shard.head_sizes == HEAD_SIZES and shard.cols_per_rank == 3


def test_session_requires_attention_before_engram(world):
    session = EngramSession(world(), max_tokens_proposal=8)
    assert (
        session.num_attention_ranks,
        session.num_engram_ranks,
        session.role_rank,
    ) == (2, 2, 0)
    assert session.metadata.tolist() == [1, 4, 0, 2, 2, 2, 0, 0]

    engram = EngramSession(
        world(ClusterType.ENGRAM, role_rank=1), max_tokens_proposal=0
    )
    assert engram.metadata.tolist() == [1, 4, 3, 2, 2, 2, 2, 1]

    with pytest.raises(ValueError, match="must not propose"):
        EngramSession(world(ClusterType.ENGRAM), max_tokens_proposal=4)


def test_session_coexists_with_af_ep_ranks(world):
    """A + F + E in one world: the Engram pair skips the F ranks and the
    metadata counts only the attention and Engram ranks."""
    from vllm_xcpu_plugin.af_ep.common.session_v8 import AfV8Session

    # Attention 0-1, MoE 2-3, Engram 4-5.
    f_ranks = MpiCluster(1, ClusterType.MOE, (2, 3))
    attention_world = world(extra_clusters=(f_ranks,), num_engram_ranks=2)
    attention_world.clusters[2] = MpiCluster(2, ClusterType.ENGRAM, (4, 5))
    session = EngramSession(attention_world, max_tokens_proposal=8)
    assert session.metadata.tolist() == [1, 4, 0, 2, 2, 2, 0, 0]

    engram_world = world(
        ClusterType.ENGRAM, extra_clusters=(f_ranks,), num_engram_ranks=2, role_rank=1
    )
    engram_world.clusters[2] = MpiCluster(2, ClusterType.ENGRAM, (4, 5))
    engram_world.global_world_comm.Get_rank = lambda: 5
    pair = engram_world.global_world_comm.Split
    engram_world.global_world_comm.Split = lambda color, key: SimpleNamespace(
        Get_rank=lambda: 3, Get_size=lambda: 4, py2f=lambda: 9
    )
    del pair
    session = EngramSession(engram_world, max_tokens_proposal=0)
    assert session.metadata.tolist() == [1, 4, 3, 2, 2, 2, 2, 1]

    # The AF-EP pair of the same attention world ignores the Engram ranks.
    attention_world.global_world_comm.Split = lambda color, key: SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 4,
        py2f=lambda: 11,
        allgather=lambda v: [v, v, 0, 0],
    )
    af = AfV8Session(attention_world, max_rows_per_attention_rank=8)
    assert af.metadata.tolist() == [2, 4, 0, 2, 2, 2, 0, 0]
    assert af.communicator_handle.item() == 11


def test_session_rejects_engram_ranks_before_attention(world):
    bad = world()
    bad.clusters[0] = MpiCluster(0, ClusterType.ATTN, (2, 3))
    bad.clusters[2] = MpiCluster(2, ClusterType.ENGRAM, (0, 1))
    bad.global_world_comm.Get_rank = lambda: 2
    with pytest.raises(ValueError, match="before all ENGRAM ranks"):
        EngramSession(bad, max_tokens_proposal=8)


def test_client_requests_the_local_window_and_collects_once(world, fake_remote_ops):
    client = RemoteEngramClient(EngramSession(world(), max_tokens_proposal=8))
    client._tp = (2, 1)  # sequence-parallel shard of TP rank 1 of 2
    client.bind_layer(True, 0)
    client.bind_layer(True, 1)
    with pytest.raises(RuntimeError, match="disagree on sequence parallelism"):
        client.bind_layer(False, 1)
    client.initialize(make_shard())
    assert fake_remote_ops[:2] == [
        "sync",
        ("initialize", 8, 2, 24, 12, [sum(HEAD_SIZES[0]), sum(HEAD_SIZES[1])]),
    ]

    assert client.local_window(5) == (3, 2)
    assert client.local_window(1) == (1, 0)  # empty shard is still a request
    hashes = torch.zeros(5, 2, 24, dtype=torch.int32)
    rows = torch.empty(3, 24, 256, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="before any request"):
        client.fill_rows(0, rows)
    client.request(hashes)
    with pytest.raises(RuntimeError, match="never collected"):
        client.request(hashes)
    with pytest.raises(RuntimeError, match="layer 0 must collect"):
        client.fill_rows(1, rows)
    client.fill_rows(0, rows)
    client.fill_rows(1, rows)
    with pytest.raises(RuntimeError, match="already collected"):
        client.fill_rows(0, rows)
    kinds = [call[0] for call in fake_remote_ops[2:]]
    assert kinds == ["request", "collect", "decode", "decode"]
    assert fake_remote_ops[2][2:4] == (3, 2)  # token_start, num_tokens
    assert fake_remote_ops[4][3:5] == (0, 2) and fake_remote_ops[5][3:5] == (1, 2)

    client.stop()
    tail = [c if c == "sync" else c[0] for c in fake_remote_ops[-4:]]
    assert tail == ["sync", "stop", "sync", "cleanup"]


def test_client_rejects_capacity_overflow(world, fake_remote_ops):
    client = RemoteEngramClient(EngramSession(world(), max_tokens_proposal=4))
    client.bind_layer(False, 0)
    client.bind_layer(False, 1)
    client.initialize(make_shard())
    with pytest.raises(RuntimeError, match="exceed the negotiated capacity"):
        client.request(torch.zeros(5, 2, 24, dtype=torch.int32))


def test_client_rejects_calls_from_compiled_graphs(world, fake_remote_ops, monkeypatch):
    """The exchange ops post / wait MPI collectives and have no fake impls;
    a traced caller must fail with the reason, not a Dynamo fake error."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    client = RemoteEngramClient(EngramSession(world(), max_tokens_proposal=8))
    client.bind_layer(False, 0)
    client.bind_layer(False, 1)
    client.initialize(make_shard())
    with pytest.raises(RuntimeError, match="outside compiled graphs"):
        client.request(torch.zeros(1, 2, 24, dtype=torch.int32))
    rows = torch.empty(1, 24, 256, dtype=torch.bfloat16)
    client._pending_tokens = 1  # past the request guard
    with pytest.raises(RuntimeError, match="outside compiled graphs"):
        client.fill_rows(0, rows)
    with pytest.raises(RuntimeError, match="outside compiled graphs"):
        client.stop()
    assert fake_remote_ops[2:] == []  # no native exchange call went through


def test_client_stop_before_initialize_aborts_the_world(world, fake_remote_ops):
    """No negotiated capacity means no stop request can be formed; the
    Engram ranks are stuck in the initialization collective, so stopping
    aborts the MPI world (once, even if shutdown runs twice)."""
    session = EngramSession(world(), max_tokens_proposal=8)
    aborted = []
    session.pair_comm.Abort = lambda code: aborted.append(code)
    client = RemoteEngramClient(session)
    client.stop()
    client.stop()
    assert aborted == [1]
    # A client that did initialize still stops through the native op.
    assert fake_remote_ops == []


def test_stop_finalizes_the_session_for_a_clean_reinitialize(world, fake_remote_ops):
    """stop() tears down the native endpoint context; the session's
    negotiated cache goes with it, so a later initialize() renegotiates
    instead of answering from stale state against a cleaned-up backend."""
    client = RemoteEngramClient(EngramSession(world(), max_tokens_proposal=8))
    client.bind_layer(False, 0)
    client.bind_layer(False, 1)
    client.initialize(make_shard())
    client.stop()
    with pytest.raises(RuntimeError, match="has not been initialized"):
        client.session.require_initialized()

    client.initialize(make_shard())  # renegotiates through the native op
    initializes = [call for call in fake_remote_ops if call[0] == "initialize"]
    assert len(initializes) == 2

    client.request(torch.zeros(1, 2, 24, dtype=torch.int32))
    client.fill_rows(0, torch.empty(1, 24, 256, dtype=torch.bfloat16))
    client.stop()  # the renegotiated session can stop again
    tail = [call if call == "sync" else call[0] for call in fake_remote_ops[-4:]]
    assert tail == ["sync", "stop", "sync", "cleanup"]
