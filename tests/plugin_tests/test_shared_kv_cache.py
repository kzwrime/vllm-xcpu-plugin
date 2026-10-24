"""Exercise real shared cache writes with two independent MCPU workers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "tp,group,packed,compiled,default_file",
    [(tp, tp, packed, False, False) for tp in (1, 2, 4) for packed in (False, True)]
    + [
        (2, 2, False, True, False),
        (4, 2, False, False, False),
        (4, 2, True, False, False),
        (2, 2, False, False, True),
    ],
)
def test_shared_cache_two_tp_pairs(tmp_path, packed, tp, group, compiled, default_file):
    processes = []
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TORCH_MCPU_SHARED_SIZE="164352",
        TORCH_MCPU_SHARED_SIM_FILE=str(tmp_path / "cache"),
        SHARED_TEST_PACKED=str(int(packed)),
        SHARED_TEST_TP=str(tp),
        SHARED_TEST_COMPILED=str(int(compiled)),
        SHARED_TEST_OUT=str(tmp_path),
        VLLM_XCPU_SHARED_KV_CACHE="1",
        VLLM_XCPU_SHARED_KV_GROUP_SIZE=str(group),
    )
    env.pop("TORCH_MCPU_SHARED_BASE", None)
    if default_file:
        env.pop("TORCH_MCPU_SHARED_SIM_FILE")
    try:
        for rank in range(2 * tp):
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
        paths = set()
        for process in processes:
            out, err = process.communicate(timeout=180)
            if process.returncode:
                failures.append(out + err)
            paths.update(
                line.removeprefix("SHARED_KV_FILE=")
                for line in out.splitlines()
                if line.startswith("SHARED_KV_FILE=")
            )
        assert not failures, "\n".join(failures)
        # One file per domain, removed once mapped.
        prefix = "/dev/shm/xcpu_shared_kv." if default_file else str(tmp_path / "cache")
        assert len(paths) == 2 * (tp // group)
        assert all(p.startswith(prefix) and not os.path.exists(p) for p in paths)
        pools = {}
        for file in tmp_path.glob("pool.*"):
            pools.setdefault(file.name.split(".")[1], set()).add(file.read_bytes())
        # Every rank of one DP rank sees identical replicas; DP ranks differ.
        assert sorted(pools) == ["dp0", "dp1"]
        assert all(len(contents) == 1 for contents in pools.values())
        assert pools["dp0"] != pools["dp1"]
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
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheLayout,
        KVCacheTensor,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    from vllm_xcpu_plugin import shared_kv_cache as shared
    from vllm_xcpu_plugin.attention import sparse_attn_indexer as indexer
    from vllm_xcpu_plugin.attention.sparse_mla import _xcpu_do_kv_cache_update

    tp = int(os.environ["SHARED_TEST_TP"])
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2 * tp,
        timeout=timedelta(seconds=45),
    )
    ranks = [list(range(i * tp, (i + 1) * tp)) for i in range(2)]
    groups = [dist.new_group(r) for r in ranks]
    device_groups = [dist.new_group(r, backend="mcpu") for r in ranks]
    group = groups[rank // tp]
    device_group = device_groups[rank // tp]
    coordinator = SimpleNamespace(
        world_size=tp,
        rank_in_group=rank % tp,
        cpu_group=group,
        barrier=lambda: dist.barrier(group=group),
        all_reduce=lambda t: dist.all_reduce(t, group=device_group),
    )

    vllm.distributed.get_tp_group = lambda: coordinator
    cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                2 * 64 * (576 * 2 + 132), ["mla"], 2 * 64 * 576 * 2, 64 * 576 * 2
            ),
            KVCacheTensor(
                2 * 64 * (576 * 2 + 132),
                ["indexer"],
                2 * 64 * 132,
                64 * 132,
                offset=2 * 64 * 576 * 2,
            ),
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
            KVCacheTensor(
                2 * stride,
                ["mla"],
                layer_stride=64 * 576 * 2,
                offset=0,
                block_stride=stride,
            ),
            KVCacheTensor(
                2 * stride,
                ["indexer"],
                layer_stride=64 * 132,
                offset=64 * 576 * 2,
                block_stride=stride,
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
    cache_config.kv_cache_tensors.append(
        KVCacheTensor(
            cache_config.kv_cache_tensors[0].size,
            ["draft"],
            layer_stride=2 * 64 * 2 * 16 * 2,
            block_stride=64 * 2 * 16 * 2,
        )
    )
    cache_config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["draft"],
            FullAttentionSpec(
                block_size=64, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
            ),
        )
    )
    runtime = shared.SharedKVCache(
        SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=rank // tp)),
        torch.device("mcpu:0"),
        draft_layers=["draft"],
    )
    caches = runtime.allocate(cache_config, torch.device("mcpu:0"), KVCacheLayout.LBNHC)
    mla = caches["mla"].view(torch.bfloat16).squeeze(1)
    idx = caches["indexer"].view(torch.uint8).squeeze(1)
    assert mla.untyped_storage().data_ptr() == idx.untyped_storage().data_ptr()
    draft = caches["draft"]
    draft.fill_(rank + 1)
    runtime.fence()
    assert torch.all(draft.cpu() == rank + 1)
    assert draft.untyped_storage().data_ptr() != mla.untyped_storage().data_ptr()
    expected_mla = torch.zeros_like(mla)
    expected_idx = torch.zeros_like(idx)
    torch.manual_seed(45 + rank // tp)
    q = torch.randn(1, 32, 128).bfloat16().to("mcpu")
    k = torch.randn(1, 128).bfloat16().to("mcpu")
    weight = torch.ones(128, device="mcpu")
    bias = torch.zeros(128, device="mcpu")
    weights = torch.ones(1, 32, device="mcpu")
    cos_sin = torch.randn(32, 64).bfloat16().to("mcpu")
    slots = torch.zeros(1, dtype=torch.int64, device="mcpu")
    positions = torch.zeros(1, dtype=torch.int64, device="mcpu")
    context = SimpleNamespace(attn_metadata={}, slot_mapping={"indexer": slots})
    indexer.get_forward_context = lambda: context
    k_cache = indexer._indexer_k_cache
    qk_cache = indexer._indexer_qk_cache
    if os.environ["SHARED_TEST_COMPILED"] == "1":
        torch._inductor.config.enable_auto_functionalized_v2 = False
        k_cache = torch.compile(k_cache, fullgraph=True)
        qk_cache = torch.compile(qk_cache, fullgraph=True)
    for step in range(8):
        # Step start: every rank zeroes the recycled block, and the queued fence
        # keeps peers from appending tokens until all ranks have zeroed it.
        for cache in (mla, idx, expected_mla, expected_idx):
            cache[1].zero_()
        runtime.device_fence()
        # Every rank in a group writes the same bytes and immediately reads them
        # back while its peers write the same slots; no fence separates the two.
        slots.fill_(64 + step % 3)
        positions.fill_(step)
        value = 10 + rank // tp * 100 + step
        kv = torch.full((1, 512), value, dtype=torch.bfloat16, device="mcpu")
        pe = torch.full((1, 1, 64), value + 1, dtype=torch.bfloat16, device="mcpu")
        torch_xcpu.ops.reshape_and_cache(
            kv, pe.squeeze(1), expected_mla, slots, kv_cache_dtype="auto"
        )
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
            q_result, weights_result = qk_cache(
                q,
                weights,
                k,
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
            k_cache(
                k,
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
        # Stand-in for the TP all-reduce that follows each rank's last cache read;
        # the next step zeroes these slots.
        runtime.device_fence()
    print(f"SHARED_KV_FILE={runtime.path}")
    pool = runtime.backing[0].cpu().numpy().tobytes()
    out = Path(os.environ["SHARED_TEST_OUT"])
    (out / f"pool.dp{rank // tp}.rank{rank}").write_bytes(pool)
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "value,expected", [("", 4), ("1", 1), ("2", 2), ("4", 4), ("3", None), ("0", None)]
)
def test_group_size_must_divide_tp(monkeypatch, value, expected):
    from vllm_xcpu_plugin.shared_kv_cache import group_size

    monkeypatch.setenv("VLLM_XCPU_SHARED_KV_GROUP_SIZE", value)
    if expected is None:
        with pytest.raises(ValueError, match="must divide TP size 4"):
            group_size(4)
    else:
        assert group_size(4) == expected


def test_domain_probe_rejects_mismatched_mapping():
    from vllm_xcpu_plugin.shared_kv_cache import check_domains

    a, b, unset = b"a" * 8, b"b" * 8, b"\0" * 8
    check_domains([(0, a, a), (0, None, a), (1, b, b), (1, None, b)])
    for records in (
        # Domain 1's second rank does not map its owner's pool.
        [(0, a, a), (0, None, a), (1, b, b), (1, None, unset)],
        # Both domains map one pool, so domain 0 reads domain 1's nonce.
        [(0, a, b), (0, None, b), (1, b, b), (1, None, b)],
    ):
        with pytest.raises(RuntimeError, match="does not match the physical"):
            check_domains(records)


@pytest.mark.parametrize("zero,copies", [([], []), ([5], []), ([], [(2, 5)])])
def test_v2_fences_after_steps_that_rewrite_blocks(monkeypatch, zero, copies):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from vllm_xcpu_plugin import shared_kv_cache
    from vllm_xcpu_plugin.worker.model_runner import GPUModelRunnerV2, McpuModelRunnerV2

    @dataclass
    class Update:
        new_block_ids_to_zero: list
        kv_cache_block_copies: list

    output = Update(zero, copies)
    calls = []
    monkeypatch.setattr(
        shared_kv_cache,
        "_runtime",
        SimpleNamespace(
            fence=lambda: calls.append("fence"),
            device_fence=lambda: calls.append("device_fence"),
        ),
    )
    monkeypatch.setattr(
        GPUModelRunnerV2, "update_requests", lambda self, x: calls.append(x)
    )
    runner = McpuModelRunnerV2.__new__(McpuModelRunnerV2)
    runner.update_requests(output)
    # Zeroing and copies run unchanged on every rank, shared layers included, and
    # only a queued fence follows them; the host never waits.
    assert calls == ([output, "device_fence"] if zero or copies else [output])


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2])
