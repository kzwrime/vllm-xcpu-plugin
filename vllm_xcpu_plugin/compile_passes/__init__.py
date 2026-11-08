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

import itertools
import logging
import os
from pathlib import Path

from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from vllm_xcpu_plugin import envs

logger = logging.getLogger(__name__)

_dump_counter = itertools.count()


def pass_uuid() -> str:
    """Hash of every pass source file; the cache-key identity of the passes."""
    files = sorted(str(p) for p in Path(__file__).parent.rglob("*.py"))
    return get_hash_for_files(tuple(files)).hex()


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
        from vllm_xcpu_plugin.compile_passes.deepseek_v32 import (
            fused_mla_qnorm_qb_uk_rope_cache as fused_pass,
        )

        _dump_graph("pre", graph)
        replaced = fused_pass.replace_with_fused_mla_qnorm_qb_uk_rope_cache(graph)
        if replaced:
            logger.info(
                "XCPU post-grad pass fused %d MLA front halves", replaced
            )
            _dump_graph("post", graph)

    def uuid(self) -> str:
        return pass_uuid()
