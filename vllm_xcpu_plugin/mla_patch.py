# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast


def maybe_patch_vllm_mla_attention() -> None:
    from vllm.model_executor.layers import attention

    from vllm_xcpu_plugin.attn_backend import XcpuTritonMLAAttention

    attention_any = cast(Any, attention)
    if getattr(attention_any, "_xcpu_mla_attention_patched", False):
        return
    attention_any.MLAAttention = XcpuTritonMLAAttention
    attention_any._xcpu_mla_attention_patched = True
