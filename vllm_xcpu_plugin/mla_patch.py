# SPDX-License-Identifier: Apache-2.0


def maybe_patch_vllm_mla_attention() -> None:
    from vllm.model_executor.layers import attention

    from vllm_xcpu_plugin.attn_backend import XcpuTritonMLAAttention

    if getattr(attention, "_xcpu_mla_attention_patched", False):
        return

    attention.MLAAttention = XcpuTritonMLAAttention
    attention._xcpu_mla_attention_patched = True
