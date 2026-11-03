# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 config rules: the indexer K cache is the MXFP4 record."""

from types import SimpleNamespace

import pytest
from vllm.config.attention import AttentionConfig

from vllm_xcpu_plugin.models.deepseek_v41.config import update_deepseek_v41_config


def _config(model_type="deepseek_v41", **attention):
    return SimpleNamespace(
        engram_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        attention_config=AttentionConfig(**attention),
    )


def test_unset_indexer_dtype_resolves_to_mxfp4():
    config = _config()
    update_deepseek_v41_config(config)
    assert config.attention_config.indexer_kv_dtype == "mxfp4"


def test_other_indexer_dtype_is_rejected():
    with pytest.raises(ValueError, match="MXFP4 key record"):
        update_deepseek_v41_config(_config(indexer_kv_dtype="fp8"))


def test_sparse_logits_is_rejected():
    with pytest.raises(ValueError, match="indexer_sparse_logits"):
        update_deepseek_v41_config(
            _config(indexer_kv_dtype="mxfp4", indexer_sparse_logits=True)
        )


def test_other_models_are_untouched():
    config = _config(model_type="qwen3")
    update_deepseek_v41_config(config)
    assert config.attention_config.indexer_kv_dtype == "auto"
