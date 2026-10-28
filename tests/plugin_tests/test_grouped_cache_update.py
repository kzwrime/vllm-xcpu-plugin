from __future__ import annotations

import pytest
import torch
import torch_xcpu  # noqa: F401

from vllm_xcpu_plugin import attn_backend
from vllm_xcpu_plugin.attn_backend import _can_group_cache_updates

LAYERS, TOKENS, HEADS, HEAD_DIM, BLOCKS, BLOCK = 3, 5, 2, 64, 4, 16


def _inputs(caches=None):
    keys = torch.randn(LAYERS, TOKENS, HEADS, HEAD_DIM, device="mcpu")
    values = torch.randn_like(keys)
    if caches is None:
        caches = [
            torch.empty(BLOCKS, HEADS, BLOCK, 2 * HEAD_DIM, device="mcpu")
            for _ in range(LAYERS)
        ]
    slots = [torch.arange(TOKENS, dtype=torch.int64, device="mcpu") for _ in caches]
    return keys, values, caches, slots


def test_disjoint_caches_are_grouped_and_cached():
    keys, values, caches, slots = _inputs()
    assert _can_group_cache_updates(keys, values, caches, slots)
    cached = attn_backend._cache_geometry
    assert _can_group_cache_updates(*_inputs(caches))
    assert attn_backend._cache_geometry is cached
    # Rebound caches are checked again.
    assert _can_group_cache_updates(*_inputs())
    assert attn_backend._cache_geometry is not cached


def test_interleaved_pages_of_one_pool_are_rejected():
    pool = torch.empty(BLOCKS * 2, HEADS, BLOCK, 2 * HEAD_DIM, device="mcpu")
    # Two views that share pages are not disjoint.
    caches = [pool[0:BLOCKS], pool[1 : BLOCKS + 1], pool[BLOCKS:]]
    assert not _can_group_cache_updates(*_inputs(caches))


def test_interleaved_disjoint_pages_are_grouped():
    pool = torch.empty(BLOCKS * LAYERS, HEADS, BLOCK, 2 * HEAD_DIM, device="mcpu")
    caches = [pool[i::LAYERS] for i in range(LAYERS)]
    assert _can_group_cache_updates(*_inputs(caches))


def test_per_call_sources_are_checked_against_cached_caches():
    keys, values, caches, slots = _inputs()
    assert _can_group_cache_updates(keys, values, caches, slots)
    # A source in a cache storage must not be grouped, even with a cache hit.
    aliased = caches[0].view(-1)[: keys.numel()].view(keys.shape)
    assert not _can_group_cache_updates(aliased, values, caches, slots)
    bad_slots = [s.int() for s in slots]
    assert not _can_group_cache_updates(keys, values, caches, bad_slots)
    assert not _can_group_cache_updates(keys, values.double(), caches, slots)


def test_cache_layout_mismatch_is_rejected():
    keys, values, caches, slots = _inputs()
    wrong = [torch.empty(BLOCKS, HEADS, BLOCK, HEAD_DIM, device="mcpu") for _ in caches]
    assert not _can_group_cache_updates(keys, values, wrong, slots)
    assert _can_group_cache_updates(keys, values, caches, slots)


def test_slot_count_must_match_caches():
    keys, values, caches, slots = _inputs()
    with pytest.raises(ValueError):
        _can_group_cache_updates(keys, values, caches, slots[:-1])
