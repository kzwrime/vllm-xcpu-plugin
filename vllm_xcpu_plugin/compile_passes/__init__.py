# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XCPU post-grad FX graph passes for MCPU torch.compile graphs.

MCPU compiles with Dynamo TRACE_ONCE + Inductor, so vLLM's PostGradPassManager
never runs; the plugin installs XcpuPostGradPass directly as
``inductor_compile_config["post_grad_custom_post_pass"]`` (see platform.py).
Inductor calls it with the post-grad ``torch.fx.Graph`` after AOTAutograd,
before its reinplace/auto-functionalize passes, so the graph still shows
``auto_functionalized`` wrappers around the mutating torch_xcpu operators.

The pass instance exposes its source-file hash as ``uuid()``. Both Inductor's
FX-graph cache key and vLLM's ``CompilationConfig.compute_hash()`` (via
config.utils.normalize_value, which folds uuid() results in and rejects plain
callables) pick that hash up, so editing any pass source invalidates both
cache layers without a separate marker string.
"""

import hashlib
import itertools
import logging
import os
from pathlib import Path

from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from vllm_xcpu_plugin import envs

logger = logging.getLogger(__name__)

_dump_counter = itertools.count()


def _active_pass_flags() -> tuple[bool, bool, bool]:
    return (
        bool(
            envs.VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE
        ),
        bool(
            envs.VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE
        ),
        bool(envs.VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM),
    )


def pass_uuid() -> str:
    """Hash of every pass source file plus active pass switches."""
    files = sorted(str(p) for p in Path(__file__).parent.rglob("*.py"))
    digest = hashlib.sha256(get_hash_for_files(tuple(files)))
    digest.update(repr(_active_pass_flags()).encode("utf-8"))
    return digest.hexdigest()


def _dump_graph(tag: str, graph) -> None:
    base = envs.VLLM_XCPU_DUMP_POST_GRAD_GRAPH
    if not base:
        return
    path = f"{base}.{os.getpid()}.{next(_dump_counter)}.{tag}.txt"
    module = getattr(graph, "owning_module", None)
    text = (
        module.print_readable(
            print_output=True, include_stride=True, include_device=True
        )
        if module is not None
        else str(graph)
    )
    with open(path, "w") as f:
        f.write(text)
    logger.info("XCPU post-grad graph dumped to %s", path)


class XcpuPostGradPass(CustomGraphPass):
    def __call__(self, graph) -> None:
        fuse_add_norm, fuse_qnorm, fuse_router_gemm = _active_pass_flags()
        if not (fuse_add_norm or fuse_qnorm or fuse_router_gemm):
            return

        _dump_graph("pre", graph)
        mla_replaced = 0
        # The MLA add-norm pass runs first: each layer it fuses removes the
        # qnorm pass's window (the first layer, with its plain rms_norm
        # input, is left for the qnorm pass).
        if fuse_add_norm:
            from vllm_xcpu_plugin.compile_passes.deepseek_v32 import (
                fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache as add_norm_pass,
            )

            mla_replaced += (
                add_norm_pass.replace_with_fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache(  # noqa: E501
                    graph
                )
            )
        if fuse_qnorm:
            from vllm_xcpu_plugin.compile_passes.deepseek_v32 import (
                fused_mla_qnorm_qb_uk_rope_cache as qnorm_pass,
            )

            mla_replaced += (
                qnorm_pass.replace_with_fused_mla_qnorm_qb_uk_rope_cache(graph)
            )
        if mla_replaced:
            logger.info(
                "XCPU post-grad pass fused %d MLA front halves", mla_replaced
            )

        router_replaced = 0
        if fuse_router_gemm:
            from vllm_xcpu_plugin.compile_passes import (
                fused_add_norm_router_gemm as router_gemm_pass,
            )

            router_replaced = (
                router_gemm_pass.replace_with_fused_add_norm_router_gemm(graph)
            )
            if router_replaced:
                logger.info(
                    "XCPU post-grad pass fused %d MoE add-norm + router GEMM "
                    "front(s)",
                    router_replaced,
                )

        if mla_replaced or router_replaced:
            _dump_graph("post", graph)

    def uuid(self) -> str:
        return pass_uuid()
