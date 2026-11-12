"""The attention loader skips the tables by name; the Engram rank loads its slice."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_xcpu_plugin.engram_disagg.attn.model_loader import (
    EngramAttentionModelLoader,
    is_engram_table,
    shards_with_engram_tables,
    skip_engram_tables_iterator,
)
from vllm_xcpu_plugin.engram_disagg.common import EngramShardLayout
from vllm_xcpu_plugin.engram_disagg.server.tables import (
    DUMMY_SCALE_BYTE,
    DUMMY_WEIGHT_BYTE,
    load_engram_shard,
)

HEAD_SIZES = tuple(tuple(3 + c for c in range(24)) for _ in range(2))


def make_shard(num_engram_ranks=2):
    return EngramShardLayout(
        layer_ids=(1, 2),
        num_embeddings=(sum(HEAD_SIZES[0]), sum(HEAD_SIZES[1])),
        head_sizes=HEAD_SIZES,
        num_engram_ranks=num_engram_ranks,
    )


@pytest.fixture
def checkpoint(tmp_path):
    """Two shards: one with a table next to small engram tensors, one without."""
    shard = make_shard()
    tables = {}
    for layer_id, rows in zip(shard.layer_ids, shard.num_embeddings):
        weight = (
            torch.arange(rows * 256, dtype=torch.int32).remainder(251).to(torch.uint8)
        )
        tables[f"layers.{layer_id}.engram.embed.weight"] = weight.view(rows, 256).view(
            torch.float8_e4m3fn
        )
        tables[f"layers.{layer_id}.engram.embed.scale"] = (
            (torch.arange(rows * 8, dtype=torch.int32).remainder(7) + 120)
            .to(torch.uint8)
            .view(rows, 8)
            .view(torch.float8_e8m0fnu)
        )
    small = {
        "layers.1.engram.q_weight": torch.ones(4, 16, dtype=torch.bfloat16),
        "layers.1.engram.wkv.weight": torch.ones(8, 16, dtype=torch.float8_e4m3fn),
        "layers.0.attn.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    shard_a = {k: v for k, v in tables.items() if ".1." in k}
    shard_a.update({k: v for k, v in small.items() if k.startswith("layers.1")})
    shard_b = {k: v for k, v in tables.items() if ".2." in k}
    shard_b["layers.0.attn.weight"] = small["layers.0.attn.weight"]
    save_file(shard_a, tmp_path / "model-00001-of-00002.safetensors")
    save_file(shard_b, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard_a}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard_b})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (tmp_path / "config.json").write_text("{}")
    return tmp_path, shard, tables


def test_table_names_and_shards(checkpoint):
    folder, _, _ = checkpoint
    assert is_engram_table("layers.14.engram.embed.weight")
    assert is_engram_table("model.layers.1.engram.embed.scale")
    assert not is_engram_table("layers.1.engram.wkv.weight")
    assert shards_with_engram_tables(str(folder), "model.safetensors.index.json") == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_attention_iterator_skips_tables_but_keeps_neighbours(checkpoint):
    folder, _, _ = checkpoint
    names = sorted(
        name
        for name, _ in skip_engram_tables_iterator([
            str(folder / "model-00001-of-00002.safetensors")
        ])
    )
    assert names == ["layers.1.engram.q_weight", "layers.1.engram.wkv.weight"]


def test_attention_iterator_filters_foreign_experts(tmp_path):
    """The table shards carry their expert neighbours through the same
    expert filter as the stock iterator: foreign experts never reach the
    loader, local ones and their scales do."""
    tensors = {
        "layers.1.engram.embed.weight": torch.ones(4, 256, dtype=torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.ones(4, 8, dtype=torch.uint8),
        "layers.1.mlp.experts.3.gate_up_proj.weight": torch.ones(2),
        "layers.1.mlp.experts.3.gate_up_proj.scale": torch.ones(1),
        "layers.1.mlp.experts.7.down_proj.weight": torch.ones(2),
        "layers.1.attn.weight": torch.ones(2),
    }
    shard_file = tmp_path / "model-00001-of-00001.safetensors"
    save_file(tensors, shard_file)

    names = sorted(
        name
        for name, _ in skip_engram_tables_iterator(
            [str(shard_file)], local_expert_ids={3}
        )
    )
    assert names == [
        "layers.1.attn.weight",
        "layers.1.mlp.experts.3.gate_up_proj.scale",
        "layers.1.mlp.experts.3.gate_up_proj.weight",
    ]
    # Without EP filtering nothing but the tables is skipped.
    names = sorted(name for name, _ in skip_engram_tables_iterator([str(shard_file)]))
    assert "layers.1.mlp.experts.7.down_proj.weight" in names


def test_attention_loader_yields_everything_but_the_tables(checkpoint):
    from vllm.config.load import LoadConfig

    folder, _, _ = checkpoint
    loader = EngramAttentionModelLoader(LoadConfig(load_format="xcpu_engram_auto"))
    source = EngramAttentionModelLoader.Source(str(folder), None)
    names = sorted(name for name, _ in loader._get_weights_iterator(source))
    assert names == [
        "layers.0.attn.weight",
        "layers.1.engram.q_weight",
        "layers.1.engram.wkv.weight",
    ]


def test_engram_rank_loads_its_column_slice(checkpoint):
    folder, shard, tables = checkpoint
    loaded = load_engram_shard(
        shard, 1, model=str(folder), revision=None, load_format="auto"
    )
    for layer_index, layer_id in enumerate(shard.layer_ids):
        start, end = shard.column_range(layer_index, 1)
        assert loaded.vocab_starts[layer_index] == start
        weight = tables[f"layers.{layer_id}.engram.embed.weight"].view(torch.uint8)
        scale = tables[f"layers.{layer_id}.engram.embed.scale"].view(torch.uint8)
        assert torch.equal(loaded.weights[layer_index], weight[start:end])
        assert torch.equal(loaded.scales[layer_index], scale[start:end])
        assert loaded.weights[layer_index].dtype == torch.uint8


def test_dummy_tables_match_vllm_dummy_values():
    shard = make_shard(num_engram_ranks=3)
    loaded = load_engram_shard(
        shard, 2, model="unused", revision=None, load_format="dummy"
    )
    start, end = shard.column_range(0, 2)
    assert loaded.weights[0].shape == (end - start, 256)
    assert (loaded.weights[0] == DUMMY_WEIGHT_BYTE).all()
    assert (loaded.scales[0] == DUMMY_SCALE_BYTE).all()
    assert (
        torch
        .tensor(DUMMY_WEIGHT_BYTE, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .item()
        == 1.0
    )


def test_plugin_entry_point_registers_the_loader_in_a_fresh_process():
    """Workers resolve `xcpu_engram_<fmt>` after the plugin's register_ops only
    (the model config is updated in another process), so the registration has
    to happen on the plugin entry point, not on the config path."""
    import subprocess
    import sys

    code = (
        "import vllm_xcpu_plugin\n"
        "vllm_xcpu_plugin.register_ops()\n"
        "from vllm.config import LoadConfig\n"
        "from vllm.model_executor.model_loader import get_model_loader\n"
        "for fmt in ('xcpu_engram_auto', 'xcpu_engram_safetensors'):\n"
        "    loader = get_model_loader(LoadConfig(load_format=fmt))\n"
        "    assert type(loader).__name__ == 'EngramAttentionModelLoader', fmt\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")
