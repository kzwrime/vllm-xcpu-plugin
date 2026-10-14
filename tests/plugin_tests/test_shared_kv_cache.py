"""Exercise real shared cache writes with two independent MCPU workers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("writer", [False, True])
def test_only_owner_mutates_cache_during_state_update(monkeypatch, writer):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from vllm_xcpu_plugin import shared_kv_cache
    from vllm_xcpu_plugin.worker.model_runner import GPUModelRunner, McpuModelRunner

    @dataclass
    class SchedulerOutput:
        new_block_ids_to_zero: list | None
        kv_cache_block_copies: list
        block_tables: list

    output = SchedulerOutput([5], [(2, 5)], [7, 5])
    calls = []

    def update_states(self, received):
        calls.append(received)

    monkeypatch.setattr(GPUModelRunner, "_update_states", update_states)
    monkeypatch.setattr(
        shared_kv_cache,
        "_runtime",
        SimpleNamespace(writer=writer, fence=lambda: calls.append("fence")),
    )
    runner = McpuModelRunner.__new__(McpuModelRunner)
    runner._update_states(output)
    assert calls[0] == calls[2] == "fence"
    received = calls[1]
    assert received.block_tables is output.block_tables
    assert received.new_block_ids_to_zero == ([5] if writer else None)
    assert received.kv_cache_block_copies == ([(2, 5)] if writer else [])
    assert output.new_block_ids_to_zero == [5]
    assert output.kv_cache_block_copies == [(2, 5)]


@pytest.mark.parametrize("packed", [False, True])
def test_shared_cache_two_tp_pairs(tmp_path, packed):
    processes = []
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TORCH_MCPU_SHARED_SIZE="262144",
        TORCH_MCPU_SHARED_SIM_FILE=str(tmp_path / "cache"),
        SHARED_TEST_PACKED=str(int(packed)),
        VLLM_XCPU_SHARED_KV_CACHE="1",
    )
    env.pop("TORCH_MCPU_SHARED_BASE", None)
    try:
        for rank in range(4):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(rank),
                        str(tmp_path / "rendezvous"),
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        failures = []
        for process in processes:
            out, err = process.communicate(timeout=90)
            if process.returncode:
                failures.append(out + err)
        assert not failures, "\n".join(failures)
        files = sorted(tmp_path.glob("cache.dp*"))
        assert len(files) == 2
        assert files[0].read_bytes() != files[1].read_bytes()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


def _worker(rank, rendezvous):
    from datetime import timedelta
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    import torch_xcpu
    import vllm.distributed
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    from vllm_xcpu_plugin import shared_kv_cache as shared
    from vllm_xcpu_plugin.flashattn_mla_sparse_patch import _xcpu_do_kv_cache_update
    from vllm_xcpu_plugin.layers import sparse_attn_indexer as indexer

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = groups[rank // 2]
    coordinator = SimpleNamespace(
        world_size=2,
        rank_in_group=rank % 2,
        cpu_group=group,
        barrier=lambda: dist.barrier(group=group),
    )

    def broadcast_object(value, src=0):
        values = [value]
        dist.broadcast_object_list(values, src=(rank // 2) * 2 + src, group=group)
        return values[0]

    coordinator.broadcast_object = broadcast_object
    vllm.distributed.get_tp_group = lambda: coordinator
    cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(2 * 64 * 576 * 2, ["mla"]),
            KVCacheTensor(2 * 64 * 132, ["indexer"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["mla"],
                MLAAttentionSpec(
                    block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
                ),
            ),
            KVCacheGroupSpec(
                ["indexer"],
                MLAAttentionSpec(
                    block_size=64, num_kv_heads=1, head_size=132, dtype=torch.uint8
                ),
            ),
        ],
    )
    packed = os.environ["SHARED_TEST_PACKED"] == "1"
    if packed:
        stride = 64 * (576 * 2 + 132)
        cache_config.kv_cache_tensors = [
            KVCacheTensor(2 * stride, ["mla"], offset=0, block_stride=stride),
            KVCacheTensor(
                2 * stride, ["indexer"], offset=64 * 576 * 2, block_stride=stride
            ),
        ]
        cache_config.kv_cache_groups = [
            KVCacheGroupSpec(
                ["mla", "indexer"],
                UniformTypeKVCacheSpecs(
                    block_size=64,
                    kv_cache_specs={
                        g.layer_names[0]: g.kv_cache_spec
                        for g in cache_config.kv_cache_groups
                    },
                ),
            )
        ]
    runtime = shared.SharedKVCache(
        SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=rank // 2)),
        torch.device("mcpu:0"),
    )
    raw = runtime.allocate(cache_config)
    if packed:
        assert raw["mla"] is raw["indexer"]
        mla = (
            raw["mla"]
            .view(torch.bfloat16)
            .as_strided((2, 64, 576), (stride // 2, 576, 1))
        )
        idx = (
            raw["indexer"]
            .view(torch.uint8)
            .as_strided((2, 64, 132), (stride, 132, 1), storage_offset=64 * 576 * 2)
        )
    else:
        mla = raw["mla"].view(torch.bfloat16).view(2, 64, 576)
        idx = raw["indexer"].view(torch.uint8).view(2, 64, 132)
    expected_mla = torch.zeros_like(mla)
    expected_idx = torch.zeros_like(idx)
    torch.manual_seed(45 + rank // 2)
    q = torch.randn(1, 32, 128).bfloat16().to("mcpu")
    k = torch.randn(1, 128).bfloat16().to("mcpu")
    weight = torch.ones(128, device="mcpu")
    bias = torch.zeros(128, device="mcpu")
    weights = torch.ones(1, 32, device="mcpu")
    cos_sin = torch.randn(32, 64).bfloat16().to("mcpu")
    slots = torch.zeros(1, dtype=torch.int64, device="mcpu")
    positions = torch.zeros(1, dtype=torch.int64, device="mcpu")
    indexer.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={}, slot_mapping={"indexer": slots}
    )
    for step in range(8):
        runtime.fence()
        # Recycle and overwrite slots while the other rank reads the same pool.
        slots.fill_(64 + step % 3)
        positions.fill_(step)
        value = 10 + rank // 2 * 100 + step
        kv = torch.full((1, 512), value, dtype=torch.bfloat16, device="mcpu")
        pe = torch.full((1, 1, 64), value + 1, dtype=torch.bfloat16, device="mcpu")
        torch_xcpu.ops.reshape_and_cache(
            kv, pe.squeeze(1), expected_mla, slots, kv_cache_dtype="auto"
        )
        # Deliberately different follower input catches accidental duplicate writes.
        if not runtime.writer:
            kv.fill_(-500)
            pe.fill_(-500)
        _xcpu_do_kv_cache_update(None, kv, pe, mla, slots, "auto", None)
        torch.testing.assert_close(mla.cpu(), expected_mla.cpu(), rtol=0, atol=0)
        if step % 2:
            q_ref, weights_ref = torch_xcpu.ops.fused_indexer_qk_rope_quant_cache(
                positions,
                q,
                cos_sin,
                weights,
                k,
                weight,
                bias,
                slots,
                expected_idx,
                1e-6,
                1.0,
                1.0,
                False,
            )
            q_result, weights_result = indexer._indexer_qk_cache(
                q,
                weights,
                k if runtime.writer else k + 10,
                weight,
                bias,
                positions,
                cos_sin,
                idx,
                "indexer",
                1e-6,
                1.0,
                1.0,
                False,
            )
            torch.testing.assert_close(q_result.cpu(), q_ref.cpu())
            torch.testing.assert_close(weights_result.cpu(), weights_ref.cpu())
        else:
            torch_xcpu.ops.fused_indexer_k_norm_rope_cache(
                k, weight, bias, positions, cos_sin, slots, expected_idx, 1e-6, False
            )
            indexer._indexer_k_cache(
                k if runtime.writer else k + 10,
                weight,
                bias,
                positions,
                cos_sin,
                idx,
                "indexer",
                1e-6,
                False,
            )
        torch.testing.assert_close(idx.cpu(), expected_idx.cpu(), rtol=0, atol=0)
        runtime.fence()
    dist.destroy_process_group()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2])
