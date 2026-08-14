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
    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    verify_upstream_compatibility(("attention",))
    import vllm_xcpu_plugin.attn_backend  # noqa


def register_ops():
    logger.info("register_ops")
    from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

    register_vllm_kernels()
    import vllm_xcpu_plugin.custom_ops  # noqa
    from vllm_xcpu_plugin.dflash_patch import maybe_patch_dflash_inputs
    from vllm_xcpu_plugin.gdn_patch import maybe_patch_gdn_attention
    from vllm_xcpu_plugin.layers.fp8_linear import register_fp8_linear_kernel
    from vllm_xcpu_plugin.layers.mxfp4_linear import register_mxfp4_linear_kernel
    from vllm_xcpu_plugin.layers.quark_mxfp4 import (
        register_quark_mxfp4_linear_scheme,
    )
    import vllm_xcpu_plugin.layers.fused_moe.prepare_finalize_factory  # noqa: F401
    import vllm_xcpu_plugin.layers.fused_moe.routed_experts  # noqa: F401
    from vllm_xcpu_plugin.fake_triton.runtime import KernelVersionError

    def install_optional_integration(name, install):
        try:
            install()
        except (ImportError, AttributeError, KernelVersionError) as exc:
            logger.warning(
                "Skipping optional %s integration because its vLLM compatibility "
                "targets are unavailable or changed: %s",
                name,
                exc,
            )

    # Model-specific patches must not block unrelated dense-model startup.
    # Their own compatibility checks still prevent stale replacements from
    # being installed for a model that needs them.
    install_optional_integration("GDN", maybe_patch_gdn_attention)
    install_optional_integration("DFlash", maybe_patch_dflash_inputs)
    install_optional_integration("FP8 linear", register_fp8_linear_kernel)
    install_optional_integration("MXFP4 linear", register_mxfp4_linear_kernel)
    install_optional_integration(
        "Quark MXFP4 W4A16 linear", register_quark_mxfp4_linear_scheme
    )
    import vllm_xcpu_plugin.layers.layernorm  # noqa
    import vllm_xcpu_plugin.layers.rotary_embedding  # noqa
    import vllm_xcpu_plugin.layers.sparse_attn_indexer  # noqa
    import vllm_xcpu_plugin.layers.qwen_gdn_linear_attn  # noqa
    import vllm_xcpu_plugin.layers.fused_moe.moe_runner  # noqa
    import vllm_xcpu_plugin.layers.fused_moe.unquantized_fused_moe_method  # noqa
    import vllm_xcpu_plugin.topk_patch as topk_patch
    import vllm_xcpu_plugin.sampler_patch as sampler_patch
    import vllm_xcpu_plugin.grouped_topk_patch as grouped_topk_patch
    # import vllm_xcpu_plugin.mla_patch as mla_patch
    import vllm_xcpu_plugin.flashattn_mla_sparse_patch as flashattn_mla_sparse_patch

    install_optional_integration(
        "MoE topk_softmax", topk_patch.maybe_patch_vllm_topk_softmax
    )
    topk_patch.maybe_patch_vllm_topk_topp_sampler()
    sampler_patch.maybe_patch_vllm_temperature()
    sampler_patch.maybe_patch_vllm_gumbel_sample()
    install_optional_integration(
        "MoE grouped_topk", grouped_topk_patch.maybe_patch_vllm_grouped_topk
    )
    # install_optional_integration(
    #     "MLA attention", mla_patch.maybe_patch_vllm_mla_attention
    # )
    install_optional_integration(
        "flashattn_mla_sparse",
        flashattn_mla_sparse_patch.maybe_patch_vllm_flashattn_mla_sparse,
    )
