"""Exercise real shared cache writes with two independent MCPU workers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "tp,packed,compiled",
    [(tp, packed, False) for tp in (1, 2, 4) for packed in (False, True)]
    + [(2, False, True)],
)
def test_shared_cache_two_tp_pairs(tmp_path, packed, tp, compiled):
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
        VLLM_XCPU_SHARED_KV_CACHE="1",
    )
    env.pop("TORCH_MCPU_SHARED_BASE", None)
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
        for process in processes:
            out, err = process.communicate(timeout=180)
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
    groups = [dist.new_group(list(range(i * tp, (i + 1) * tp))) for i in range(2)]
    group = groups[rank // tp]
    coordinator = SimpleNamespace(
        world_size=tp,
        rank_in_group=rank % tp,
        cpu_group=group,
        barrier=lambda: dist.barrier(group=group),
    )

    vllm.distributed.get_tp_group = lambda: coordinator
    vllm.distributed.get_pcp_group = lambda: SimpleNamespace(rank_in_group=0)
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
        runtime.fence()
        # Recycle and overwrite slots while the other rank reads the same pool.
        slots.fill_(64 + step % 3)
        positions.fill_(step)
        value = 10 + rank // tp * 100 + step
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
            q_result, weights_result = qk_cache(
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
            k_cache(
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


@pytest.mark.parametrize("writer", [False, True])
def test_v2_updates_private_draft_and_only_owner_updates_shared(monkeypatch, writer):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from vllm.v1.worker import utils

    from vllm_xcpu_plugin import shared_kv_cache
    from vllm_xcpu_plugin.worker.model_runner import GPUModelRunnerV2, McpuModelRunnerV2

    @dataclass
    class Update:
        new_block_ids_to_zero: list
        kv_cache_block_copies: list
        block_table_updates: dict

    output = Update([5], [(2, 5)], {"request": [7, 5]})
    calls = []
    shared = SimpleNamespace(
        writer=writer, private_caches=["draft"], fence=lambda: calls.append("fence")
    )
    monkeypatch.setattr(shared_kv_cache, "_runtime", shared)
    monkeypatch.setattr(
        GPUModelRunnerV2, "update_requests", lambda self, x: calls.append(x)
    )
    monkeypatch.setattr(
        utils,
        "copy_kv_cache_blocks_inplace",
        lambda caches, blocks, copies: calls.append((caches, copies)),
    )
    runner = McpuModelRunnerV2.__new__(McpuModelRunnerV2)
    runner.kv_caches = ["target", "draft"]
    runner.kv_cache_config = SimpleNamespace(num_blocks=8)
    runner.update_requests(output)
    assert calls[0] == calls[-1] == "fence"
    assert calls[1].block_table_updates is output.block_table_updates
    assert calls[1].new_block_ids_to_zero == [5]
    assert calls[1].kv_cache_block_copies == []
    assert calls[2] == (["target", "draft"] if writer else ["draft"], [(2, 5)])
    assert output.kv_cache_block_copies == [(2, 5)]


@pytest.mark.parametrize("writer", [False, True])
def test_v2_zeroing_keeps_private_draft_on_followers(monkeypatch, writer):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from vllm.v1.worker import utils

    from vllm_xcpu_plugin import shared_kv_cache
    from vllm_xcpu_plugin.worker.model_runner import GPUModelRunnerV2, McpuModelRunnerV2

    @dataclass
    class Group:
        layer_names: list

    captured = []
    monkeypatch.setattr(
        shared_kv_cache,
        "_runtime",
        SimpleNamespace(writer=writer, draft_layers={"draft"}),
    )
    monkeypatch.setattr(
        GPUModelRunnerV2, "_init_kv_zero_meta", lambda self: captured.append("all")
    )
    monkeypatch.setattr(
        utils,
        "KVBlockZeroer",
        lambda *args, **kwargs: captured.extend(
            n for g in kwargs["attn_groups_iter"] for n in g.layer_names
        ),
    )
    runner = McpuModelRunnerV2.__new__(McpuModelRunnerV2)
    runner.device = None
    runner.attn_groups = [[Group(["target"]), Group(["draft"])]]
    runner.kernel_block_sizes = [64, 64]
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(num_blocks=8)
    runner._init_kv_zero_meta()
    assert captured == (["all"] if writer else ["draft"])


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2])
