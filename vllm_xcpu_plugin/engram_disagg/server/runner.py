"""Engram-rank lifecycle: join the MPI world, load the shard, serve, exit.

Python only bootstraps: after ``deepseek_v41_engram_serve`` starts, every
step runs inside that native call until all attention ranks send the stop
request. The process then exits without an MPI collective (the attention
ranks may already have been terminated by the launcher).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngramServerOptions:
    model: str
    revision: str | None = None
    load_format: str = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.load_format not in ("auto", "safetensors", "dummy"):
            raise ValueError(f"unsupported Engram load format: {self.load_format}")


def run_engram_server(options: EngramServerOptions) -> None:
    import mpi4py

    from vllm_xcpu_plugin.disagg import run_role_service
    from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

    # Exit without MPI_Finalize: the attention ranks never finalize and may
    # be gone by the time the stop request has been served.
    mpi4py.rc.finalize = False
    run_role_service(
        ClusterType.ENGRAM,
        lambda mpi_world: _load(options, mpi_world),
        lambda state: _serve(*state),
        finalize=False,
    )


def _load(options: EngramServerOptions, mpi_world):
    from vllm.models.deepseek_v41.common.engram import EngramLayout
    from vllm.transformers_utils.config import get_config, get_hf_text_config

    from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

    from ..common import EngramSession, EngramShardLayout
    from .tables import load_engram_shard

    assert mpi_world.cluster_type == ClusterType.ENGRAM
    session = EngramSession(mpi_world, max_tokens_proposal=0)
    # The engram fields sit on the text config of the VL checkpoint.
    text_config = get_hf_text_config(
        get_config(options.model, options.trust_remote_code, options.revision)
    )
    layout = EngramLayout.from_config(text_config)
    if layout is None:
        raise ValueError(f"{options.model} declares no engram layers")
    shard = EngramShardLayout.from_engram_layout(layout, session.num_engram_ranks)
    tables = load_engram_shard(
        shard,
        session.role_rank,
        model=options.model,
        revision=options.revision,
        load_format=options.load_format,
    )
    print(
        f"Engram rank {session.role_rank} tables ready: "
        + ", ".join(
            f"layer {layer} rows [{start}, {start + weight.shape[0]})"
            for layer, weight, start in zip(
                shard.layer_ids, tables.weights, tables.vocab_starts
            )
        ),
        flush=True,
    )
    return session, shard, tables


def _serve(session, shard, tables) -> None:
    import torch
    from torch_xcpu import ops as xcpu_ops

    max_tokens = session.initialize(shard)
    slots = torch.zeros(
        session.num_attention_ranks,
        xcpu_ops.deepseek_v41_engram_remote_slot_ints(
            max_tokens, shard.layers, shard.cols_per_rank
        ),
        dtype=torch.int32,
        device="cpu",
    )
    send = torch.zeros(
        session.num_attention_ranks,
        xcpu_ops.deepseek_v41_engram_remote_slot_bytes(
            max_tokens, shard.layers, shard.cols_per_rank
        ),
        dtype=torch.uint8,
        device="cpu",
    )
    print(f"Engram rank {session.role_rank} serving", flush=True)
    xcpu_ops.deepseek_v41_engram_serve(
        tables.weights,
        tables.scales,
        tables.vocab_starts,
        slots,
        send,
        session.metadata,
        session.communicator_handle,
    )
    xcpu_ops.deepseek_v41_engram_remote_cleanup()
    print(f"Engram rank {session.role_rank} stopped", flush=True)
