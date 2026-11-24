# SPDX-License-Identifier: Apache-2.0
"""Multi-rank end-to-end EngramParallel model test over synthetic safetensors.

Verifies the full non-dummy model path across 4 MPI ranks (DP=2, TP=2, SP=True):
EngramParallelModelLoader.get_all_weights -> _make_deepseek_v4_weights_mapper ->
_engram_parallel_weight_loader -> XcpuEngram.forward, comparing against an
unsharded full-table reference bit for bit.

Run with:
  mpirun --allow-run-as-root --bind-to none -np 4 \
    python tests/mpi/engram_parallel_model_mpi.py
"""

# MPI must be initialized before importing torch_mcpu / vllm.
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from types import SimpleNamespace

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
if provided != MPI.THREAD_MULTIPLE:
    raise RuntimeError(f"Expected MPI_THREAD_MULTIPLE, got {provided}")

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ["VLLM_CPU_USE_MPI"] = "1"
os.environ["VLLM_XCPU_ENABLE_ENGRAM_PARALLEL"] = "0"
os.environ["VLLM_XCPU_ENABLE_ENGRAM_DISAGG"] = "0"
os.environ["TORCH_XCPU_ENABLE_CHECK"] = "1"

import torch
import torch.nn.functional as F
import torch_mcpu  # noqa: F401
import torch_xcpu
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from vllm.config import (
    EngramConfig,
    LoadConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.deepseek_v41.common.engram import DEAD_ID, EngramLayout
from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper

import vllm_xcpu_plugin
from vllm_xcpu_plugin.distributed.mpi_world import (
    ClusterType,
    initialize_mpi_world,
)
from vllm_xcpu_plugin.engram_disagg.attn.runtime import unregister_engram_client
from vllm_xcpu_plugin.engram_parallel.bootstrap import (
    bootstrap_engram_parallel_client,
)
from vllm_xcpu_plugin.engram_parallel.model_loader import (
    EngramParallelModelLoader,
    configure_engram_parallel_model_loader,
)
from vllm_xcpu_plugin.models.deepseek_v41.engram import XcpuEngram

DIM = 256
HIDDEN_SIZE = 5120
HC_MULT = 4
LAYER_IDS = (1, 3)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_hf_text_config() -> SimpleNamespace:
    # Build primes first with minimal bucket geometry so total rows stay compact.
    probe = SimpleNamespace(
        engram_layer_ids=list(LAYER_IDS),
        engram_num_embeddings=[10_000, 10_000],
        engram_max_ngram_size=4,
        engram_vocab_size=30,
        engram_n_heads=8,
        engram_head_dim=DIM,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=64,
    )
    layout = EngramLayout(probe)
    active_per_layer = [
        sum(size for ngram in layer for size in ngram) for layer in layout.primes
    ]
    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        hc_mult=HC_MULT,
        rms_norm_eps=1e-6,
        engram_layer_ids=list(LAYER_IDS),
        engram_num_embeddings=[active_per_layer[0] + 11, active_per_layer[1] + 13],
        engram_max_ngram_size=4,
        engram_vocab_size=30,
        engram_n_heads=8,
        engram_head_dim=DIM,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=64,
    )


def _make_synthetic_checkpoint(
    ckpt_dir: str, hf_text_config: SimpleNamespace
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(20261123)
    tensors: dict[str, torch.Tensor] = {}

    for idx, lid in enumerate(LAYER_IDS):
        n_rows = hf_text_config.engram_num_embeddings[idx]
        # Distinct non-zero FP8 values and UE8M0 scales per (layer, row, col).
        row_ids = torch.arange(n_rows, dtype=torch.int32).unsqueeze(1)
        col_ids = torch.arange(DIM, dtype=torch.int32).unsqueeze(0)
        raw_vals = ((row_ids * (17 + idx * 13) + col_ids * 7 + 3) % 120 + 1).to(
            torch.float32
        ) / 16.0
        sign = torch.where((row_ids + col_ids + idx) % 3 == 0, -1.0, 1.0)
        tensors[f"layers.{lid}.engram.embed.weight"] = (raw_vals * sign).to(
            torch.float8_e4m3fn
        )
        scale_bytes = (
            (row_ids * (5 + idx) + torch.arange(DIM // 32).unsqueeze(0)) % 5 + 125
        ).to(torch.uint8)
        tensors[f"layers.{lid}.engram.embed.scale"] = scale_bytes.view(
            torch.float8_e8m0fnu
        )
        u = (
            torch.randn(
                (HC_MULT + 1) * HIDDEN_SIZE, 1, generator=g, dtype=torch.float32
            )
            * 0.05
        )
        v = torch.randn(1, 24 * DIM, generator=g, dtype=torch.float32) * 0.25
        tensors[f"layers.{lid}.engram.wkv.weight"] = (
            (u * v).to(torch.bfloat16).contiguous()
        )
        tensors[f"layers.{lid}.engram.q_weight"] = (
            torch.randn(HC_MULT, HIDDEN_SIZE, generator=g, dtype=torch.float32) * 0.25
        ).to(torch.bfloat16)
        tensors[f"layers.{lid}.engram.k_weight"] = (
            torch.randn(HC_MULT, HIDDEN_SIZE, generator=g, dtype=torch.float32) * 0.25
        ).to(torch.bfloat16)

    tensors["norm.weight"] = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16)

    shard1_keys = [
        f"layers.{LAYER_IDS[0]}.engram.embed.weight",
        f"layers.{LAYER_IDS[0]}.engram.embed.scale",
        f"layers.{LAYER_IDS[0]}.engram.wkv.weight",
    ]
    shard2_keys = [
        f"layers.{LAYER_IDS[1]}.engram.embed.weight",
        f"layers.{LAYER_IDS[1]}.engram.embed.scale",
        f"layers.{LAYER_IDS[1]}.engram.q_weight",
        f"layers.{LAYER_IDS[1]}.engram.k_weight",
    ]
    shard3_keys = [
        f"layers.{LAYER_IDS[0]}.engram.q_weight",
        f"layers.{LAYER_IDS[0]}.engram.k_weight",
        f"layers.{LAYER_IDS[1]}.engram.wkv.weight",
        "norm.weight",
    ]
    shards = [
        ("model-00001-of-00003.safetensors", shard1_keys),
        ("model-00002-of-00003.safetensors", shard2_keys),
        ("model-00003-of-00003.safetensors", shard3_keys),
    ]
    weight_map: dict[str, str] = {}
    for fname, keys in shards:
        save_file({k: tensors[k] for k in keys}, os.path.join(ckpt_dir, fname))
        for k in keys:
            weight_map[k] = fname

    with open(
        os.path.join(ckpt_dir, "model.safetensors.index.json"), "w"
    ) as handle:
        json.dump({"metadata": {}, "weight_map": weight_map}, handle)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as handle:
        json.dump({"architectures": ["DeepseekV41ForCausalLM"]}, handle)
    return tensors


class _MiniNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )


class _MiniBackbone(nn.Module):
    def __init__(
        self, hf_text_config: SimpleNamespace, engram_layout: EngramLayout
    ) -> None:
        super().__init__()
        self.norm = _MiniBackbone._make_norm(hf_text_config.hidden_size)
        layers: dict[str, nn.Module] = {}
        for idx, lid in enumerate(LAYER_IDS):
            block = nn.Module()
            block.engram = XcpuEngram(
                config=hf_text_config,
                layout=engram_layout,
                layer_hash_index=idx,
                quant_config=None,
                prefix=f"model.layers.{lid}.engram",
                use_sequence_parallel=True,
            )
            layers[str(lid)] = block
        self.layers = nn.ModuleDict(layers)

    @staticmethod
    def _make_norm(hidden_size: int) -> _MiniNorm:
        return _MiniNorm(hidden_size)


class _MiniCausalLM(nn.Module):
    def __init__(
        self, hf_text_config: SimpleNamespace, engram_layout: EngramLayout
    ) -> None:
        super().__init__()
        self.model = _MiniBackbone(hf_text_config, engram_layout)


def _reference_layer_forward(
    *,
    full_tensors: dict[str, torch.Tensor],
    layer_id: int,
    num_embeddings: int,
    hidden_sp: torch.Tensor,
    hash_layer: torch.Tensor,
    token_mask: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    num_tokens = hash_layer.shape[0]
    chunk = (num_tokens + tp_size - 1) // tp_size
    start = min(tp_rank * chunk, num_tokens)
    local_tokens = min(chunk, num_tokens - start)

    emb_out = torch.zeros(
        chunk, hash_layer.shape[1], DIM, dtype=torch.bfloat16, device="mcpu"
    )
    if num_tokens > 0:
        full_emb = torch.zeros(
            num_tokens, hash_layer.shape[1], DIM, dtype=torch.bfloat16, device="mcpu"
        )
        weight_mcpu = full_tensors[f"layers.{layer_id}.engram.embed.weight"].to("mcpu")
        scale_mcpu = (
            full_tensors[f"layers.{layer_id}.engram.embed.scale"]
            .view(torch.uint8)
            .to("mcpu")
        )
        torch_xcpu.ops.deepseek_v41_engram_lookup(
            weight_mcpu,
            scale_mcpu,
            hash_layer.to("mcpu"),
            full_emb,
            0,
            num_embeddings,
            0,
            hash_layer.shape[1],
        )
        if local_tokens > 0:
            emb_out[:local_tokens].copy_(full_emb[start : start + local_tokens])

    wkv_w = full_tensors[f"layers.{layer_id}.engram.wkv.weight"].to("mcpu")
    kv = F.linear(emb_out.flatten(-2), wkv_w)
    q_w = full_tensors[f"layers.{layer_id}.engram.q_weight"].to("mcpu")
    k_w = full_tensors[f"layers.{layer_id}.engram.k_weight"].to("mcpu")
    local_mask = token_mask[start : start + local_tokens].to("mcpu")
    return torch_xcpu.ops.deepseek_v41_engram_post_wkv(
        hidden_sp, kv, q_w, k_w, local_mask, local_tokens, 1e-6, 1e-6
    )


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if world_size != 4:
        raise RuntimeError(f"Expected 4 MPI ranks, got {world_size}")

    tp_size = 2
    dp_size = 2
    hf_text_config = _make_hf_text_config()

    ckpt_dir = tempfile.mkdtemp(prefix="engram_par_model_") if rank == 0 else None
    master_port = _find_free_port() if rank == 0 else None
    ckpt_dir, master_port = comm.bcast((ckpt_dir, master_port), root=0)
    assert isinstance(ckpt_dir, str) and isinstance(master_port, int)

    if rank == 0:
        _make_synthetic_checkpoint(ckpt_dir, hf_text_config)
    comm.Barrier()

    full_tensors: dict[str, torch.Tensor] = {}
    for fname in (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ):
        with safe_open(os.path.join(ckpt_dir, fname), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118
                full_tensors[key] = handle.get_tensor(key)

    os.environ.update({
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })

    vllm_xcpu_plugin.xcpu_platform_plugin()
    vllm_xcpu_plugin.register_ops()

    load_cfg = LoadConfig(load_format="auto")
    configure_engram_parallel_model_loader(load_cfg)
    parallel_cfg = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
    )
    tp_rank_expected = rank % tp_size
    dp_rank_expected = rank // tp_size
    assert parallel_cfg.data_parallel_rank == dp_rank_expected
    del tp_rank_expected, dp_rank_expected
    sched_cfg = SchedulerConfig(
        max_num_batched_tokens=32,
        max_num_seqs=16,
        max_model_len=64,
        is_encoder_decoder=False,
    )
    vllm_config = VllmConfig(
        parallel_config=parallel_cfg,
        load_config=load_cfg,
        scheduler_config=sched_cfg,
    )
    vllm_config.engram_config = EngramConfig()
    os.environ["VLLM_XCPU_ENABLE_ENGRAM_PARALLEL"] = "1"

    bootstrap_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_text_config=hf_text_config,
        ),
        parallel_config=parallel_cfg,
        engram_config=SimpleNamespace(embedding_across_dp=False),
        load_config=load_cfg,
        scheduler_config=sched_cfg,
    )
    mpi_world = initialize_mpi_world(ClusterType.ATTN)
    torch_xcpu.initialize_runtime()

    client = None
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{master_port}",
                local_rank=rank,
                backend="gloo",
            )
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=1,
                backend="gloo",
            )
            tp_rank = get_tp_group().rank_in_group
            dp_rank = rank // tp_size

            client = bootstrap_engram_parallel_client(
                mpi_world=mpi_world,
                vllm_config=bootstrap_cfg,
                world_rank=rank,
                world_size=world_size,
            )
            engram_layout = EngramLayout(hf_text_config)
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device("mcpu"):
                    model = _MiniCausalLM(hf_text_config, engram_layout)
            finally:
                torch.set_default_dtype(prev_dtype)

            loader = get_model_loader(load_cfg)
            assert isinstance(loader, EngramParallelModelLoader)

            mapper = _make_deepseek_v4_weights_mapper("fp4")
            raw_weights = loader.get_all_weights(
                SimpleNamespace(model=ckpt_dir, revision=None), model
            )
            params = dict(model.named_parameters())
            loaded_names: set[str] = set()
            for name, weight in mapper.apply(raw_weights):
                assert name in params, f"unexpected mapped tensor {name}"
                param = params[name]
                loader_fn = getattr(param, "weight_loader", default_weight_loader)
                loader_fn(param, weight)
                loaded_names.add(name)
            assert loaded_names == set(params.keys()), (
                f"unloaded params: {set(params.keys()) - loaded_names}"
            )
            for module in model.modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)

            client.initialize_from_engram_layout(engram_layout)

            flat_primes = [
                [p for ngram in layer for p in ngram] for layer in engram_layout.primes
            ]
            col_ranges = []
            for sizes in flat_primes:
                acc = 0
                bounds = []
                for sz in sizes:
                    bounds.append((acc, acc + sz))
                    acc += sz
                col_ranges.append(bounds)

            # Vary token counts across DP replicas and test zero-token + SP padding.
            step_tokens = [(11, 7), (0, 5), (5, 0), (0, 0), (16, 1)]
            for step, (t_dp0, t_dp1) in enumerate(step_tokens):
                num_tokens = t_dp0 if dp_rank == 0 else t_dp1
                dp_tokens = torch.tensor([t_dp0, t_dp1], dtype=torch.int32)
                chunk = (num_tokens + tp_size - 1) // tp_size
                gen = torch.Generator(device="cpu").manual_seed(
                    900 + step * 10 + dp_rank
                )

                hashes_cpu = torch.empty(num_tokens, 2, 24, dtype=torch.int32)
                for l_idx in range(2):
                    for col, (lo, hi) in enumerate(col_ranges[l_idx]):
                        hashes_cpu[:, l_idx, col] = torch.randint(
                            lo, hi, (num_tokens,), generator=gen, dtype=torch.int32
                        )
                if num_tokens > 0:
                    hashes_cpu.view(-1)[step::19] = DEAD_ID

                mask_cpu = torch.ones(num_tokens, dtype=torch.bool)
                if num_tokens > 1:
                    mask_cpu[::3] = False

                hidden_sp = (
                    torch.randn(chunk, HC_MULT, HIDDEN_SIZE, generator=gen) * 0.5
                ).to(torch.bfloat16).to("mcpu")
                hashes_mcpu = hashes_cpu.to("mcpu")
                mask_mcpu = mask_cpu.to("mcpu")

                e1 = model.model.layers["1"].engram
                e3 = model.model.layers["3"].engram
                with set_forward_context(
                    None,
                    vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=dp_tokens,
                ):
                    client.request(hashes_mcpu)
                    out1 = e1(hidden_sp, hashes_mcpu[:, 0], mask_mcpu)
                    out2 = e3(out1, hashes_mcpu[:, 1], mask_mcpu)
                torch.accelerator.synchronize()

                ref1 = _reference_layer_forward(
                    full_tensors=full_tensors,
                    layer_id=LAYER_IDS[0],
                    num_embeddings=hf_text_config.engram_num_embeddings[0],
                    hidden_sp=hidden_sp,
                    hash_layer=hashes_cpu[:, 0],
                    token_mask=mask_cpu,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
                ref2 = _reference_layer_forward(
                    full_tensors=full_tensors,
                    layer_id=LAYER_IDS[1],
                    num_embeddings=hf_text_config.engram_num_embeddings[1],
                    hidden_sp=ref1,
                    hash_layer=hashes_cpu[:, 1],
                    token_mask=mask_cpu,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
                torch.accelerator.synchronize()
                assert torch.equal(out1.cpu(), ref1.cpu()), (
                    f"rank {rank} step {step} layer 1 mismatch"
                )
                assert torch.equal(out2.cpu(), ref2.cpu()), (
                    f"rank {rank} step {step} layer 3 mismatch"
                )
                if rank == 0:
                    print(
                        f"engram_parallel_model_mpi step {step} "
                        f"tokens={(t_dp0, t_dp1)} ok",
                        flush=True,
                    )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if client is not None:
            client.stop()
            unregister_engram_client(client)
        destroy_model_parallel()
        destroy_distributed_environment()
        comm.Barrier()
        if rank == 0 and ckpt_dir is not None:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        MPI.Finalize()


if __name__ == "__main__":
    main()
