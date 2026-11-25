# SPDX-License-Identifier: Apache-2.0
"""Tests for EngramParallel 2D layout, safetensors slicing loader, and weight loader."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open as _real_safe_open
from safetensors.torch import save_file
from vllm.config.load import LoadConfig

from vllm_xcpu_plugin.engram_parallel.layout import EngramParallelLayout
from vllm_xcpu_plugin.engram_parallel.model_loader import (
    EngramParallelModelLoader,
    configure_engram_parallel_model_loader,
    slice_engram_tables_iterator,
)
from vllm_xcpu_plugin.models.deepseek_v41.engram import _engram_parallel_weight_loader


def _load_reshard_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "reshard_hf_safetensors.py"
    spec = importlib.util.spec_from_file_location("reshard_hf_safetensors", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _column_ranges(head_sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    starts = 0
    out: list[tuple[int, int]] = []
    for sz in head_sizes:
        out.append((starts, starts + sz))
        starts += sz
    return out


def _sample_sizes() -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]
]:
    layer_ids = (1, 14)
    head_sizes = (
        tuple(101 + 2 * c for c in range(24)),
        tuple(113 + 3 * c for c in range(24)),
    )
    num_embeddings = tuple(sum(s) + 17 for s in head_sizes)
    return layer_ids, num_embeddings, head_sizes


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8, 24, 32, 48, 64])
def test_layout_partitions_all_active_rows_across_world_sizes(world_size: int) -> None:
    layer_ids, num_embeddings, head_sizes = _sample_sizes()
    layout = EngramParallelLayout.from_sizes(
        layer_ids=layer_ids,
        num_embeddings=num_embeddings,
        head_sizes=head_sizes,
        world_size=world_size,
    )
    assert layout.layers == 2
    assert layout.total_cols == 24
    assert layout.rank_cuts[0] == 0
    assert layout.rank_cuts[-1] == layout.layer_base[-1]
    assert list(layout.rank_cuts) == sorted(layout.rank_cuts)
    assert len(layout.cols_per_rank) == world_size
    assert max(layout.cols_per_rank) == layout.max_cols_per_rank
    assert min(layout.cols_per_rank) >= 1

    for l_idx in range(layout.layers):
        active = layout.active_rows[l_idx]
        slices = [layout.layer_slice(l_idx, r) for r in range(world_size)]
        # Non-empty slices form an exact contiguous partition of [0, active).
        cursor = 0
        for r, (lo, hi) in enumerate(slices):
            lo2, hi2, expected_full = layout.layer_slice_by_id(layer_ids[l_idx], r)
            assert (lo, hi) == (lo2, hi2)
            assert expected_full == num_embeddings[l_idx]
            assert 0 <= lo <= hi <= active
            if hi > lo:
                assert lo == cursor
                cursor = hi
        assert cursor == active

    if world_size == 48:
        assert layout.max_cols_per_rank == 1
    elif world_size == 24:
        assert layout.max_cols_per_rank == 2
    elif world_size in (32, 64):
        assert layout.max_cols_per_rank <= 2


def test_layout_from_hf_text_config_and_validation_errors() -> None:
    hf_text_config = SimpleNamespace(
        engram_layer_ids=[1, 2],
        engram_num_embeddings=[97968, 102814],
        engram_max_ngram_size=4,
        engram_vocab_size=4000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=99092,
    )
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    upstream_layout = EngramLayout.from_config(hf_text_config)
    from_hf = EngramParallelLayout.from_hf_text_config(hf_text_config, world_size=4)
    from_up = EngramParallelLayout.from_engram_layout(upstream_layout, world_size=4)
    assert from_hf == from_up
    assert from_hf.layer_ids == (1, 2)
    assert from_hf.layers == 2
    assert from_hf.total_cols == 24

    with pytest.raises(ValueError, match="at least one engram layer"):
        EngramParallelLayout.from_sizes(
            layer_ids=(), num_embeddings=(), head_sizes=(), world_size=2
        )
    with pytest.raises(ValueError, match="world_size > 0"):
        EngramParallelLayout.from_sizes(
            layer_ids=(1,), num_embeddings=(10,), head_sizes=((10,),), world_size=0
        )
    with pytest.raises(ValueError, match="exceed its table"):
        EngramParallelLayout.from_sizes(
            layer_ids=(1,), num_embeddings=(9,), head_sizes=((10,),), world_size=1
        )


def test_slice_engram_tables_iterator_handles_empty_slices_and_neighbors(
    tmp_path,
) -> None:
    layer_id = 1
    full_w = (torch.arange(20 * 16, dtype=torch.int32).reshape(20, 16) % 7).to(
        torch.float8_e4m3fn
    )
    full_s = torch.arange(20 * 2, dtype=torch.uint8).reshape(20, 2).view(
        torch.float8_e8m0fnu
    )
    extra = torch.ones(4, 4, dtype=torch.bfloat16)
    expert0 = torch.full((2, 2), 1.0, dtype=torch.bfloat16)
    expert3 = torch.full((2, 2), 3.0, dtype=torch.bfloat16)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.1.engram.embed.weight": full_w,
            "model.layers.1.engram.embed.scale": full_s,
            "model.layers.1.engram.q_weight": extra,
            "model.layers.1.mlp.experts.0.up_proj.weight": expert0,
            "model.layers.1.mlp.experts.3.up_proj.weight": expert3,
        },
        str(shard),
    )

    # Non-empty slice + EP expert filter.
    items = dict(
        slice_engram_tables_iterator(
            [str(shard)], {layer_id: (5, 13, 20)}, local_expert_ids={0}
        )
    )
    assert items["model.layers.1.engram.embed.weight"].shape == (8, 16)
    assert torch.equal(
        items["model.layers.1.engram.embed.weight"].view(torch.uint8),
        full_w[5:13].view(torch.uint8),
    )
    assert torch.equal(
        items["model.layers.1.engram.embed.scale"].view(torch.uint8),
        full_s[5:13].view(torch.uint8),
    )
    assert torch.equal(items["model.layers.1.engram.q_weight"], extra)
    assert "model.layers.1.mlp.experts.0.up_proj.weight" in items
    assert "model.layers.1.mlp.experts.3.up_proj.weight" not in items

    # 0:0 slice on a rank owning rows only from other layers/tables.
    empty_items = dict(
        slice_engram_tables_iterator([str(shard)], {layer_id: (0, 0, 20)})
    )
    assert empty_items["model.layers.1.engram.embed.weight"].shape == (0, 16)
    assert empty_items["model.layers.1.engram.embed.scale"].shape == (0, 2)

    with pytest.raises(ValueError, match="expected 99"):
        list(slice_engram_tables_iterator([str(shard)], {layer_id: (0, 5, 99)}))
    with pytest.raises(ValueError, match="missing from EngramParallelLayout"):
        list(slice_engram_tables_iterator([str(shard)], {14: (0, 5, 20)}))

    bad_shard = tmp_path / "bad.safetensors"
    save_file({"engram.embed.weight": full_w}, str(bad_shard))
    with pytest.raises(ValueError, match="unexpected Engram table tensor name"):
        list(slice_engram_tables_iterator([str(bad_shard)], {layer_id: (0, 5, 20)}))


def test_configure_loader_and_full_model_loader_roundtrip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = LoadConfig(load_format="auto")
    configure_engram_parallel_model_loader(cfg)
    assert cfg.load_format == "xcpu_engram_parallel_auto"

    dummy_cfg = LoadConfig(load_format="dummy")
    configure_engram_parallel_model_loader(dummy_cfg)
    assert dummy_cfg.load_format == "dummy"

    with pytest.raises(ValueError, match="does not support load format"):
        configure_engram_parallel_model_loader(LoadConfig(load_format="pt"))

    # Two Engram layers across two table shards plus a non-table shard.
    layer_ids = (1, 14)
    head_sizes = ((7, 11, 13, 17), (19, 23, 29, 31))
    num_embeddings = (sum(head_sizes[0]) + 4, sum(head_sizes[1]) + 6)
    world_size = 3
    layout = EngramParallelLayout.from_sizes(
        layer_ids=layer_ids,
        num_embeddings=num_embeddings,
        head_sizes=head_sizes,
        world_size=world_size,
    )

    w1 = (torch.arange(num_embeddings[0] * 32) % 13).reshape(num_embeddings[0], 32).to(
        torch.float8_e4m3fn
    )
    s1 = (
        (torch.arange(num_embeddings[0] * 1) % 251)
        .reshape(num_embeddings[0], 1)
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    w14 = (torch.arange(num_embeddings[1] * 32) % 17).reshape(num_embeddings[1], 32).to(
        torch.float8_e4m3fn
    )
    s14 = (
        (torch.arange(num_embeddings[1] * 1) % 241)
        .reshape(num_embeddings[1], 1)
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    tok = torch.ones(4, 32, dtype=torch.bfloat16)

    save_file(
        {
            "model.layers.1.engram.embed.weight": w1,
            "model.layers.1.engram.embed.scale": s1,
        },
        str(tmp_path / "model-00001-of-00003.safetensors"),
    )
    save_file(
        {
            "model.layers.14.engram.embed.weight": w14,
            "model.layers.14.engram.embed.scale": s14,
        },
        str(tmp_path / "model-00002-of-00003.safetensors"),
    )
    save_file(
        {"model.embed_tokens.weight": tok},
        str(tmp_path / "model-00003-of-00003.safetensors"),
    )
    index = {
        "metadata": {},
        "weight_map": {
            "model.layers.1.engram.embed.weight": "model-00001-of-00003.safetensors",
            "model.layers.1.engram.embed.scale": "model-00001-of-00003.safetensors",
            "model.layers.14.engram.embed.weight": "model-00002-of-00003.safetensors",
            "model.layers.14.engram.embed.scale": "model-00002-of-00003.safetensors",
            "model.embed_tokens.weight": "model-00003-of-00003.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    loader = EngramParallelModelLoader(
        LoadConfig(load_format="xcpu_engram_parallel_auto")
    )
    source = EngramParallelModelLoader.Source(
        model_or_path=str(tmp_path), revision=None, prefix="", fall_back_to_pt=False
    )

    reconstructed_w1, reconstructed_s1 = [], []
    reconstructed_w14, reconstructed_s14 = [], []
    for rank in range(world_size):
        fake_client = SimpleNamespace(
            layout=layout, session=SimpleNamespace(world_rank=rank)
        )
        monkeypatch.setattr(
            "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
            lambda c=fake_client: c,
        )
        loaded = dict(loader._get_weights_iterator(source))
        assert torch.equal(loaded["model.embed_tokens.weight"], tok)
        reconstructed_w1.append(
            loaded["model.layers.1.engram.embed.weight"].view(torch.uint8)
        )
        reconstructed_s1.append(
            loaded["model.layers.1.engram.embed.scale"].view(torch.uint8)
        )
        reconstructed_w14.append(
            loaded["model.layers.14.engram.embed.weight"].view(torch.uint8)
        )
        reconstructed_s14.append(
            loaded["model.layers.14.engram.embed.scale"].view(torch.uint8)
        )

    assert torch.equal(
        torch.cat(reconstructed_w1, dim=0),
        w1[: layout.active_rows[0]].view(torch.uint8),
    )
    assert torch.equal(
        torch.cat(reconstructed_s1, dim=0),
        s1[: layout.active_rows[0]].view(torch.uint8),
    )
    assert torch.equal(
        torch.cat(reconstructed_w14, dim=0),
        w14[: layout.active_rows[1]].view(torch.uint8),
    )
    assert torch.equal(
        torch.cat(reconstructed_s14, dim=0),
        s14[: layout.active_rows[1]].view(torch.uint8),
    )


def test_engram_parallel_weight_loader_handles_presliced_empty_and_dummy() -> None:
    # Pre-sliced weights (weight and UE8M0 scale).
    param_w = torch.nn.Parameter(
        torch.empty(4, 16, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    param_w.engram_vocab_start = 3
    src_w = (torch.arange(4 * 16) % 5).reshape(4, 16).to(torch.float8_e4m3fn)
    _engram_parallel_weight_loader(param_w, src_w)
    assert torch.equal(param_w.data.view(torch.uint8), src_w.view(torch.uint8))

    param_s = torch.nn.Parameter(
        torch.empty(4, 2, dtype=torch.uint8), requires_grad=False
    )
    param_s.engram_vocab_start = 3
    src_s = (
        torch.arange(4 * 2, dtype=torch.uint8)
        .reshape(4, 2)
        .view(torch.float8_e8m0fnu)
    )
    _engram_parallel_weight_loader(param_s, src_s)
    assert torch.equal(param_s.data, src_s.view(torch.uint8))

    # Full un-sliced weight (e.g., DummyModelLoader): narrows [3:7].
    full_w = (torch.arange(12 * 16) % 7).reshape(12, 16).to(torch.float8_e4m3fn)
    _engram_parallel_weight_loader(param_w, full_w)
    assert torch.equal(param_w.data.view(torch.uint8), full_w[3:7].view(torch.uint8))

    # Empty 0:0 slice with either empty or full dummy tensor.
    empty_param = torch.nn.Parameter(
        torch.empty(0, 16, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    empty_param.engram_vocab_start = 0
    _engram_parallel_weight_loader(
        empty_param, torch.empty(0, 16, dtype=torch.float8_e4m3fn)
    )
    _engram_parallel_weight_loader(empty_param, full_w)

    # Shape / slice mismatches fail with ValueError (even under Python -O).
    with pytest.raises(ValueError, match="does not fit param"):
        _engram_parallel_weight_loader(
            param_w, torch.empty(12, 8, dtype=torch.float8_e4m3fn)
        )
    with pytest.raises(ValueError, match="does not fit param"):
        _engram_parallel_weight_loader(
            empty_param, torch.empty(1, 8, dtype=torch.float8_e4m3fn)
        )
    param_oob = torch.nn.Parameter(
        torch.empty(4, 16, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    param_oob.engram_vocab_start = 10
    with pytest.raises(ValueError, match="does not fit param"):
        _engram_parallel_weight_loader(param_oob, full_w)


def test_register_ops_registers_engram_parallel_loaders() -> None:
    code = """
import vllm_xcpu_plugin
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm_xcpu_plugin.engram_parallel.model_loader import EngramParallelModelLoader

vllm_xcpu_plugin.register_ops()
for fmt in ("xcpu_engram_parallel_auto", "xcpu_engram_parallel_safetensors"):
    loader = get_model_loader(LoadConfig(load_format=fmt))
    assert isinstance(loader, EngramParallelModelLoader), (
        f"unexpected loader for {fmt}: {type(loader)}"
    )
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_loader_draft_model_and_single_file_safetensors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

    class _DraftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

    def _make_engram_marker() -> torch.nn.Module:
        mod = object.__new__(ParallelEngramEmbedding)
        torch.nn.Module.__init__(mod)
        return mod

    class _TargetModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.engram = _make_engram_marker()

    loader = EngramParallelModelLoader(
        LoadConfig(load_format="xcpu_engram_parallel_auto")
    )

    # 1) Standalone draft checkpoint: single file without index, no Engram client.
    draft_dir = tmp_path / "draft_single"
    draft_dir.mkdir()
    proj_w = torch.randn(4, 4, dtype=torch.bfloat16)
    save_file({"proj.weight": proj_w}, str(draft_dir / "model.safetensors"))
    monkeypatch.setattr(
        "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
        lambda: None,
    )
    draft_cfg = SimpleNamespace(model=str(draft_dir), revision=None)
    loaded_draft = dict(loader.get_all_weights(draft_cfg, _DraftModel()))
    assert torch.equal(loaded_draft["proj.weight"], proj_w)
    assert loader._skip_engram_tables is False

    # 2) Draft model sharing bundled checkpoint containing Engram tables: skips tables.
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    w1 = torch.zeros(12, 16, dtype=torch.uint8).view(torch.float8_e4m3fn)
    s1 = torch.zeros(12, 1, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    save_file(
        {
            "model.layers.1.engram.embed.weight": w1,
            "model.layers.1.engram.embed.scale": s1,
            "proj.weight": proj_w,
        },
        str(bundled_dir / "model-00001-of-00002.safetensors"),
    )
    head_w = torch.ones(2, 4, dtype=torch.bfloat16)
    save_file(
        {"lm_head.weight": head_w},
        str(bundled_dir / "model-00002-of-00002.safetensors"),
    )
    shard1 = "model-00001-of-00002.safetensors"
    shard2 = "model-00002-of-00002.safetensors"
    (bundled_dir / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "model.layers.1.engram.embed.weight": shard1,
                "model.layers.1.engram.embed.scale": shard1,
                "proj.weight": shard1,
                "lm_head.weight": shard2,
            }
        })
    )
    bundled_cfg = SimpleNamespace(model=str(bundled_dir), revision=None)
    loaded_bundled = dict(loader.get_all_weights(bundled_cfg, _DraftModel()))
    assert set(loaded_bundled.keys()) == {"proj.weight", "lm_head.weight"}
    assert torch.equal(loaded_bundled["proj.weight"], proj_w)
    assert torch.equal(loaded_bundled["lm_head.weight"], head_w)

    # 3) Multi-file checkpoint without index reports "EngramParallel".
    no_idx_dir = tmp_path / "no_index"
    no_idx_dir.mkdir()
    save_file({"a.weight": proj_w}, str(no_idx_dir / "part1.safetensors"))
    save_file({"b.weight": head_w}, str(no_idx_dir / "part2.safetensors"))
    no_idx_cfg = SimpleNamespace(model=str(no_idx_dir), revision=None)
    with pytest.raises(ValueError, match="EngramParallel needs the safetensors index"):
        list(loader.get_all_weights(no_idx_cfg, _DraftModel()))

    # 4) Target model slices single-file checkpoint and rejects tableless ckpt.
    layout = EngramParallelLayout.from_sizes(
        layer_ids=(1,), num_embeddings=(12,), head_sizes=((5, 5),), world_size=2
    )
    fake_client = SimpleNamespace(layout=layout, session=SimpleNamespace(world_rank=0))
    monkeypatch.setattr(
        "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
        lambda: fake_client,
    )
    target_single = tmp_path / "target_single"
    target_single.mkdir()
    save_file(
        {
            "model.layers.1.engram.embed.weight": w1,
            "model.layers.1.engram.embed.scale": s1,
            "proj.weight": proj_w,
        },
        str(target_single / "model.safetensors"),
    )
    target_cfg = SimpleNamespace(model=str(target_single), revision=None)
    loaded_target = dict(loader.get_all_weights(target_cfg, _TargetModel()))
    lo, hi, _ = layout.layer_slice_by_id(1, 0)
    assert loaded_target["model.layers.1.engram.embed.weight"].shape[0] == hi - lo
    assert torch.equal(loaded_target["proj.weight"], proj_w)

    with pytest.raises(ValueError, match="expected Engram table tensors"):
        list(loader.get_all_weights(draft_cfg, _TargetModel()))


def test_engram_parallel_end_to_end_model_mpi() -> None:
    plugin_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    script = os.path.join(plugin_root, "tests", "mpi", "engram_parallel_model_mpi.py")
    # Strip any C-level PMIx/OMPI singleton variables left if an earlier pytest
    # test imported mpi4py.MPI, so the child mpirun can spawn its own 4-rank world.
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            ("PMIX_", "OMPI_COMM_", "OMPI_UNIVERSE_", "OMPI_MCA_orte_", "PMI_")
        )
    }
    proc = subprocess.run(
        [
            "mpirun",
            "--allow-run-as-root",
            "--bind-to",
            "none",
            "-np",
            "4",
            sys.executable,
            script,
        ],
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "engram_parallel_model_mpi step 2" in proc.stdout, proc.stderr


def _make_presplit_fixture_checkpoint(
    tmp_path: Path,
    *,
    engram_shard_size: int = 8192,
    target_shard_size: int = 4096,
):
    layer_ids = (1, 14)
    head_sizes = (
        tuple(19 + 2 * c for c in range(24)),
        tuple(23 + 2 * c for c in range(24)),
    )
    num_embeddings = (sum(head_sizes[0]) + 7, sum(head_sizes[1]) + 9)
    dim = 64

    w1 = (
        (torch.arange(num_embeddings[0] * dim, dtype=torch.int32) % 241)
        .to(torch.uint8)
        .reshape(num_embeddings[0], dim)
        .view(torch.float8_e4m3fn)
    )
    s1 = (
        (torch.arange(num_embeddings[0] * (dim // 32), dtype=torch.int32) % 256)
        .to(torch.uint8)
        .reshape(num_embeddings[0], dim // 32)
        .view(torch.float8_e8m0fnu)
    )
    s1[0, 0] = torch.tensor(0, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    s1[-1, -1] = torch.tensor(255, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    w14 = (
        (torch.arange(num_embeddings[1] * dim, dtype=torch.int32) % 233)
        .to(torch.uint8)
        .reshape(num_embeddings[1], dim)
        .view(torch.float8_e4m3fn)
    )
    s14 = (
        (torch.arange(num_embeddings[1] * (dim // 32), dtype=torch.int32) % 256)
        .to(torch.uint8)
        .reshape(num_embeddings[1], dim // 32)
        .view(torch.float8_e8m0fnu)
    )
    s14[1, 0] = torch.tensor(0, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    s14[-2, -1] = torch.tensor(255, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    wkv1 = torch.ones(8, dim, dtype=torch.bfloat16)
    q1 = torch.full((4, 16), 2.0, dtype=torch.bfloat16)
    k14 = torch.full((4, 16), 3.0, dtype=torch.bfloat16)
    tok = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)

    src = tmp_path / "src"
    src.mkdir()
    save_file(
        {
            "layers.1.engram.embed.weight": w1,
            "layers.1.engram.embed.scale": s1,
            "layers.1.engram.wkv.weight": wkv1,
        },
        str(src / "model-00001-of-00003.safetensors"),
    )
    save_file(
        {
            "layers.14.engram.embed.weight": w14,
            "layers.14.engram.embed.scale": s14,
            "layers.14.engram.k_weight": k14,
        },
        str(src / "model-00002-of-00003.safetensors"),
    )
    save_file(
        {
            "layers.1.engram.q_weight": q1,
            "embed_tokens.weight": tok,
        },
        str(src / "model-00003-of-00003.safetensors"),
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "layers.1.engram.embed.weight": "model-00001-of-00003.safetensors",
                "layers.1.engram.embed.scale": "model-00001-of-00003.safetensors",
                "layers.1.engram.wkv.weight": "model-00001-of-00003.safetensors",
                "layers.14.engram.embed.weight": "model-00002-of-00003.safetensors",
                "layers.14.engram.embed.scale": "model-00002-of-00003.safetensors",
                "layers.14.engram.k_weight": "model-00002-of-00003.safetensors",
                "layers.1.engram.q_weight": "model-00003-of-00003.safetensors",
                "embed_tokens.weight": "model-00003-of-00003.safetensors",
            }
        })
    )

    dst = tmp_path / "presplit"
    reshard_mod = _load_reshard_module()
    reshard_mod.reshard(
        src,
        dst,
        target_shard_size=target_shard_size,
        engram_shard_size=engram_shard_size,
    )
    orig = {
        "layers.1.engram.embed.weight": w1,
        "layers.1.engram.embed.scale": s1,
        "layers.14.engram.embed.weight": w14,
        "layers.14.engram.embed.scale": s14,
        "layers.1.engram.wkv.weight": wkv1,
        "layers.1.engram.q_weight": q1,
        "layers.14.engram.k_weight": k14,
        "embed_tokens.weight": tok,
    }
    return dst, layer_ids, num_embeddings, head_sizes, orig


@pytest.mark.parametrize("world_size", [2, 3, 4, 48])
def test_presplit_reshard_and_loader_across_world_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    dst, layer_ids, num_embeddings, head_sizes, orig = (
        _make_presplit_fixture_checkpoint(
            tmp_path, engram_shard_size=8192, target_shard_size=4096
        )
    )
    layout = EngramParallelLayout.from_sizes(
        layer_ids=layer_ids,
        num_embeddings=num_embeddings,
        head_sizes=head_sizes,
        world_size=world_size,
    )
    loader = EngramParallelModelLoader(
        LoadConfig(load_format="xcpu_engram_parallel_auto")
    )
    source = EngramParallelModelLoader.Source(
        model_or_path=str(dst), revision=None, prefix="", fall_back_to_pt=False
    )

    reconstructed = {
        "w1": [],
        "s1": [],
        "w14": [],
        "s14": [],
    }
    for rank in range(world_size):
        fake_client = SimpleNamespace(
            layout=layout, session=SimpleNamespace(world_rank=rank)
        )
        monkeypatch.setattr(
            "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
            lambda c=fake_client: c,
        )
        loaded = dict(loader._get_weights_iterator(source))
        assert torch.equal(
            loaded["layers.1.engram.wkv.weight"], orig["layers.1.engram.wkv.weight"]
        )
        assert torch.equal(
            loaded["layers.1.engram.q_weight"], orig["layers.1.engram.q_weight"]
        )
        assert torch.equal(
            loaded["layers.14.engram.k_weight"], orig["layers.14.engram.k_weight"]
        )
        assert torch.equal(loaded["embed_tokens.weight"], orig["embed_tokens.weight"])

        lo1, hi1, _ = layout.layer_slice_by_id(1, rank)
        lo14, hi14, _ = layout.layer_slice_by_id(14, rank)
        assert loaded["layers.1.engram.embed.weight"].shape == (hi1 - lo1, 64)
        assert loaded["layers.1.engram.embed.scale"].shape == (hi1 - lo1, 2)
        assert loaded["layers.14.engram.embed.weight"].shape == (hi14 - lo14, 64)
        assert loaded["layers.14.engram.embed.scale"].shape == (hi14 - lo14, 2)

        reconstructed["w1"].append(
            loaded["layers.1.engram.embed.weight"].view(torch.uint8)
        )
        reconstructed["s1"].append(
            loaded["layers.1.engram.embed.scale"].view(torch.uint8)
        )
        reconstructed["w14"].append(
            loaded["layers.14.engram.embed.weight"].view(torch.uint8)
        )
        reconstructed["s14"].append(
            loaded["layers.14.engram.embed.scale"].view(torch.uint8)
        )

    assert torch.equal(
        torch.cat(reconstructed["w1"], dim=0),
        orig["layers.1.engram.embed.weight"][: layout.active_rows[0]].view(torch.uint8),
    )
    assert torch.equal(
        torch.cat(reconstructed["s1"], dim=0),
        orig["layers.1.engram.embed.scale"][: layout.active_rows[0]].view(torch.uint8),
    )
    assert torch.equal(
        torch.cat(reconstructed["w14"], dim=0),
        orig["layers.14.engram.embed.weight"][: layout.active_rows[1]].view(
            torch.uint8
        ),
    )
    assert torch.equal(
        torch.cat(reconstructed["s14"], dim=0),
        orig["layers.14.engram.embed.scale"][: layout.active_rows[1]].view(torch.uint8),
    )


def test_presplit_loader_only_opens_overlapping_shards_and_unmaps_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

    dst, layer_ids, num_embeddings, head_sizes, _ = _make_presplit_fixture_checkpoint(
        tmp_path, engram_shard_size=4096, target_shard_size=4096
    )
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    engram_shards = {
        fname for key, fname in weight_map.items() if ".row_slice." in key
    }

    def _mapped_presplit_files() -> set[str]:
        mapped: set[str] = set()
        dst_prefix = str(dst.resolve()) + "/"
        with open("/proc/self/maps", encoding="utf-8") as proc_maps:
            for line in proc_maps:
                if dst_prefix in line:
                    path = line.strip().split()[-1]
                    mapped.add(os.path.basename(path))
        return mapped

    opened_files: list[str] = []
    max_concurrent_engram_maps = 0

    class _TrackedSafeOpen:
        def __init__(self, filename: str, *args, **kwargs):
            self._basename = os.path.basename(filename)
            opened_files.append(self._basename)
            self._inner = _real_safe_open(filename, *args, **kwargs)

        def __enter__(self):
            nonlocal max_concurrent_engram_maps
            handle = self._inner.__enter__()
            active = len(_mapped_presplit_files() & engram_shards)
            max_concurrent_engram_maps = max(max_concurrent_engram_maps, active)
            return handle

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

    monkeypatch.setattr(
        "vllm_xcpu_plugin.engram_parallel.model_loader.safe_open",
        _TrackedSafeOpen,
    )

    class _TargetModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mod = object.__new__(ParallelEngramEmbedding)
            torch.nn.Module.__init__(mod)
            self.engram = mod

    class _DraftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

    loader = EngramParallelModelLoader(
        LoadConfig(load_format="xcpu_engram_parallel_auto")
    )
    model_cfg = SimpleNamespace(model=str(dst), revision=None)

    # Check W=48 (each rank owns rows in only one layer; the other is 0:0).
    layout48 = EngramParallelLayout.from_sizes(
        layer_ids=layer_ids,
        num_embeddings=num_embeddings,
        head_sizes=head_sizes,
        world_size=48,
    )
    for rank in (0, 24, 47):
        opened_files.clear()
        max_concurrent_engram_maps = 0
        fake_client = SimpleNamespace(
            layout=layout48, session=SimpleNamespace(world_rank=rank)
        )
        monkeypatch.setattr(
            "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
            lambda c=fake_client: c,
        )
        loaded = {}
        for k, t in loader.get_all_weights(model_cfg, _TargetModel()):
            loaded[k] = t
            # After each yielded tensor, no Engram shard may remain mapped.
            assert (_mapped_presplit_files() & engram_shards) == set()
        assert (_mapped_presplit_files() & engram_shards) == set()
        assert max_concurrent_engram_maps <= 1

        # Compute exact expected overlapping Engram shards for this rank.
        expected_engram_shards: set[str] = set()
        for l_id in layer_ids:
            lo, hi, _ = layout48.layer_slice_by_id(l_id, rank)
            if lo == hi:
                continue
            for key, fname in weight_map.items():
                if f"layers.{l_id}.engram.embed." not in key:
                    continue
                s_lo, s_hi = map(int, key.rsplit(".row_slice.", 1)[1].split("_"))
                if s_hi > lo and s_lo < hi:
                    expected_engram_shards.add(fname)

        opened_engram_shards = set(opened_files) & engram_shards
        assert opened_engram_shards == expected_engram_shards
        assert len(opened_engram_shards) < len(engram_shards)
        # Empty-slice layer (0:0) with metadata opens 0 shards of that layer.
        for l_id in layer_ids:
            lo, hi, _ = layout48.layer_slice_by_id(l_id, rank)
            if lo == hi:
                layer_shards = {
                    fname
                    for k, fname in weight_map.items()
                    if f"layers.{l_id}.engram.embed." in k
                }
                assert (opened_engram_shards & layer_shards) == set()

    # Draft model skips all pure Engram shards with 0 safe_open calls on them.
    opened_files.clear()
    monkeypatch.setattr(
        "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
        lambda: None,
    )
    draft_loaded = dict(loader.get_all_weights(model_cfg, _DraftModel()))
    assert "layers.1.engram.embed.weight" not in draft_loaded
    assert "layers.1.engram.wkv.weight" in draft_loaded
    assert (set(opened_files) & engram_shards) == set()


def test_presplit_loader_without_metadata_empty_slice_and_slice_iterator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dst, layer_ids, num_embeddings, head_sizes, orig = (
        _make_presplit_fixture_checkpoint(
            tmp_path, engram_shard_size=4096, target_shard_size=4096
        )
    )
    index_path = dst / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"].pop("engram_row_slices", None)
    index_path.write_text(json.dumps(index, indent=2))

    layout48 = EngramParallelLayout.from_sizes(
        layer_ids=layer_ids,
        num_embeddings=num_embeddings,
        head_sizes=head_sizes,
        world_size=48,
    )
    fake_client = SimpleNamespace(
        layout=layout48, session=SimpleNamespace(world_rank=0)
    )
    monkeypatch.setattr(
        "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
        lambda: fake_client,
    )
    loader = EngramParallelModelLoader(
        LoadConfig(load_format="xcpu_engram_parallel_auto")
    )
    source = EngramParallelModelLoader.Source(
        model_or_path=str(dst), revision=None, prefix="", fall_back_to_pt=False
    )
    loaded = dict(loader._get_weights_iterator(source))
    assert loaded["layers.14.engram.embed.weight"].shape == (0, 64)
    assert loaded["layers.14.engram.embed.scale"].shape == (0, 2)

    # Direct slice_engram_tables_iterator call on pre-split shard files.
    all_files = sorted(str(p) for p in dst.glob("*.safetensors"))
    layer_slices = {
        1: (5, 45, num_embeddings[0]),
        14: (0, 30, num_embeddings[1]),
    }
    direct = dict(slice_engram_tables_iterator(all_files, layer_slices))
    assert torch.equal(
        direct["layers.1.engram.embed.weight"].view(torch.uint8),
        orig["layers.1.engram.embed.weight"][5:45].view(torch.uint8),
    )
    assert torch.equal(
        direct["layers.14.engram.embed.scale"].view(torch.uint8),
        orig["layers.14.engram.embed.scale"][0:30].view(torch.uint8),
    )


def test_presplit_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

    class _TargetModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mod = object.__new__(ParallelEngramEmbedding)
            torch.nn.Module.__init__(mod)
            self.engram = mod

    layout = EngramParallelLayout.from_sizes(
        layer_ids=(1,), num_embeddings=(10,), head_sizes=((4, 4),), world_size=2
    )
    fake_client = SimpleNamespace(layout=layout, session=SimpleNamespace(world_rank=0))
    monkeypatch.setattr(
        "vllm_xcpu_plugin.engram_parallel.model_loader.get_engram_client",
        lambda: fake_client,
    )
    loader = EngramParallelModelLoader(
        LoadConfig(load_format="xcpu_engram_parallel_auto")
    )

    w_part = torch.zeros(5, 16, dtype=torch.float8_e4m3fn)
    s_part = torch.zeros(5, 1, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    w_full = torch.zeros(10, 16, dtype=torch.float8_e4m3fn)
    s_full = torch.zeros(10, 1, dtype=torch.uint8)
    w_6 = torch.zeros(6, 16, dtype=torch.float8_e4m3fn)
    w_4 = torch.zeros(4, 16, dtype=torch.float8_e4m3fn)
    w_8 = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)

    def _run_case(name: str, files_spec: dict[str, dict[str, torch.Tensor]], err: str):
        case_dir = tmp_path / name
        case_dir.mkdir()
        weight_map = {}
        for fname, tensors in files_spec.items():
            save_file(
                {k: v.clone() for k, v in tensors.items()},
                str(case_dir / fname),
            )
            for k in tensors:
                weight_map[k] = fname
        (case_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        cfg = SimpleNamespace(model=str(case_dir), revision=None)
        with pytest.raises(ValueError, match=err):
            list(loader.get_all_weights(cfg, _TargetModel()))

    # 1. Start != 0
    _run_case(
        "nonzero_start",
        {
            "s1.safetensors": {"layers.1.engram.embed.weight.row_slice.5_10": w_part},
            "s2.safetensors": {
                "layers.1.engram.embed.scale.row_slice.0_5": s_part,
                "layers.1.engram.embed.scale.row_slice.5_10": s_part,
            },
        },
        "start at 5, expected 0",
    )
    # 2. Gap
    _run_case(
        "gap",
        {
            "s1.safetensors": {
                "layers.1.engram.embed.weight.row_slice.0_4": w_part[:4],
                "layers.1.engram.embed.weight.row_slice.6_10": w_part[:4],
            },
            "s2.safetensors": {"layers.1.engram.embed.scale.row_slice.0_10": s_full},
        },
        "have a gap",
    )
    # 3. Overlap
    _run_case(
        "overlap",
        {
            "s1.safetensors": {
                "layers.1.engram.embed.weight.row_slice.0_6": w_6,
                "layers.1.engram.embed.weight.row_slice.4_10": w_6,
            },
            "s2.safetensors": {"layers.1.engram.embed.scale.row_slice.0_10": s_full},
        },
        "overlap",
    )
    # 4. Total rows mismatch
    _run_case(
        "total_mismatch",
        {
            "s1.safetensors": {"layers.1.engram.embed.weight.row_slice.0_8": w_8},
            "s2.safetensors": {"layers.1.engram.embed.scale.row_slice.0_10": s_full},
        },
        "has 8 rows, expected 10",
    )
    # 5. Mixed split + unsplit
    _run_case(
        "mixed",
        {
            "s1.safetensors": {"layers.1.engram.embed.weight": w_full},
            "s2.safetensors": {"layers.1.engram.embed.weight.row_slice.0_10": w_full},
        },
        "mix split and unsplit",
    )
    # 6. Chunk shape mismatch against key range
    _run_case(
        "chunk_rows_mismatch",
        {
            "s1.safetensors": {
                "layers.1.engram.embed.weight.row_slice.0_5": w_4,
                "layers.1.engram.embed.weight.row_slice.5_10": w_part,
            },
            "s2.safetensors": {"layers.1.engram.embed.scale.row_slice.0_10": s_full},
        },
        "has shape .* expected \\(5,",
    )
    # 7. Invalid suffix
    _run_case(
        "bad_suffix",
        {"s1.safetensors": {"layers.1.engram.embed.weight.bad": w_part}},
        "unexpected Engram table tensor name",
    )


def test_vl_model_streaming_load_weights() -> None:
    from vllm.model_executor.models.utils import StageMissingLayer, WeightsMapper
    from vllm.models.deepseek_v41.nvidia.vl_model import (
        _make_deepseek_v4_vl_weights_mapper,
    )

    from vllm_xcpu_plugin.models.deepseek_v41.model import XcpuDeepseekV41ForCausalLM

    class _DummyChild(torch.nn.Module):
        def __init__(self, expected_param: str, consume_log: list[str]) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.empty(2, 2), requires_grad=False)
            self._expected_param = expected_param
            self._consume_log = consume_log

        def load_weights(self, weights):
            loaded = set()
            for name, tensor in weights:
                self._consume_log.append(f"lm:{name}")
                if name == self._expected_param:
                    self.w.data.copy_(tensor)
                    loaded.add("w")
            return loaded

    class _DummyTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Parameter(torch.empty(2, 2), requires_grad=False)

    def _build_stub(limit_mm_zero: bool, consume_log: list[str]):
        model = object.__new__(XcpuDeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            "fp4", "weight_scale"
        )
        model.language_model = _DummyChild("w", consume_log)
        model.language_model.hf_to_vllm_mapper = WeightsMapper()
        model.image_start = torch.nn.Parameter(torch.empty(4), requires_grad=False)
        model.image_end = torch.nn.Parameter(torch.empty(4), requires_grad=False)
        model.image_newline = torch.nn.Parameter(torch.empty(4), requires_grad=False)
        if limit_mm_zero:
            model.vision = StageMissingLayer("vision_tower", _DummyTower())
            model.aligner = StageMissingLayer("vision_tower", _DummyTower())
        else:
            model.vision = _DummyTower()
            model.aligner = _DummyTower()
        return model

    # Case 1: vision/aligner replaced by StageMissingLayer, image_* kept, streaming.
    events: list[str] = []
    stub_zero = _build_stub(limit_mm_zero=True, consume_log=events)
    stub_zero.language_model._expected_param = "model.norm.weight"

    def _gen_valid():
        events.clear()
        events.append("yield:vision")
        yield ("vision.proj", torch.full((2, 2), 9.0))
        events.append("yield:norm")
        yield ("norm.weight", torch.full((2, 2), 4.0))
        assert "lm:model.norm.weight" in events
        events.append("yield:image_start")
        yield ("image_start", torch.full((4,), 5.0))
        events.append("yield:image_end")
        yield ("image_end", torch.full((4,), 6.0))
        events.append("yield:image_newline")
        yield ("image_newline", torch.full((4,), 7.0))
        events.append("yield:aligner")
        yield ("aligner.proj", torch.full((2, 2), 8.0))

    loaded_zero = stub_zero.load_weights(_gen_valid())
    assert stub_zero._weights_finalized is True
    assert loaded_zero == {
        "language_model.w",
        "image_start",
        "image_end",
        "image_newline",
    }
    assert torch.equal(stub_zero.language_model.w, torch.full((2, 2), 4.0))
    assert torch.equal(stub_zero.image_start, torch.full((4,), 5.0))

    # Case 2: active vision/aligner towers also load cleanly after language_model.
    events2: list[str] = []
    stub_mm = _build_stub(limit_mm_zero=False, consume_log=events2)
    stub_mm.language_model._expected_param = "model.norm.weight"

    def _gen_mm():
        yield ("vision.proj", torch.full((2, 2), 9.0))
        yield ("norm.weight", torch.full((2, 2), 4.0))
        yield ("aligner.proj", torch.full((2, 2), 8.0))
        yield ("image_start", torch.full((4,), 5.0))

    loaded_mm = stub_mm.load_weights(_gen_mm())
    assert loaded_mm == {
        "language_model.w",
        "vision.proj",
        "aligner.proj",
        "image_start",
    }
    assert torch.equal(stub_mm.vision.proj, torch.full((2, 2), 9.0))
    assert torch.equal(stub_mm.aligner.proj, torch.full((2, 2), 8.0))
