# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

logger = logging.getLogger(__name__)


def xcpu_platform_plugin() -> str | None:
    from vllm_xcpu_plugin.fake_triton import install_fake_triton

    install_fake_triton(replace_existing=True)
    import torch_mcpu  # noqa: F401

    return "vllm_xcpu_plugin.platform.McpuPlatform"


def register_attn_backend():
    logger.info("register_attn_backend")
    import vllm_xcpu_plugin.attn_backend  # noqa


def register_ops():
    logger.info("register_ops")
    from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

    register_vllm_kernels()
    import vllm_xcpu_plugin.custom_ops  # noqa
    from vllm_xcpu_plugin.gdn_patch import maybe_patch_gdn_attention

    import vllm_xcpu_plugin.layers.layernorm  # noqa
    import vllm_xcpu_plugin.layers.rotary_embedding  # noqa
    import vllm_xcpu_plugin.layers.gdn_linear_attn  # noqa
    import vllm_xcpu_plugin.layers.fused_moe.unquantized_fused_moe_method  # noqa
    import vllm_xcpu_plugin.topk_patch as topk_patch
    import vllm_xcpu_plugin.sampler_patch as sampler_patch
    import vllm_xcpu_plugin.grouped_topk_patch as grouped_topk_patch
    import vllm_xcpu_plugin.mla_patch as mla_patch

    maybe_patch_gdn_attention()
    topk_patch.maybe_patch_vllm_topk_softmax()
    topk_patch.maybe_patch_vllm_topk_topp_sampler()
    sampler_patch.maybe_patch_vllm_temperature()
    sampler_patch.maybe_patch_vllm_gumbel_sample()
    grouped_topk_patch.maybe_patch_vllm_grouped_topk()
    mla_patch.maybe_patch_vllm_mla_attention()
