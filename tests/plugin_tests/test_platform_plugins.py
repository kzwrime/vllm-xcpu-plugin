# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.plugins import load_general_plugins

# def test_platform_plugins():
#     # simulate workload by running an example
#     import runpy
#     current_file = __file__
#     import os
#     example_file = os.path.join(
#         os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
#         "examples", "offline_inference/basic/basic.py")
#     runpy.run_path(example_file)

#     # check if the plugin is loaded correctly
#     from vllm.platforms import _init_trace, current_platform
#     assert current_platform.device_name == "DummyDevice", (
#         f"Expected DummyDevice, got {current_platform.device_name}, "
#         "possibly because current_platform is imported before the plugin"
#         f" is loaded. The first import:\n{_init_trace}")


def test_oot_custom_op(monkeypatch: pytest.MonkeyPatch):
    load_general_plugins()
    from vllm import ir

    from vllm_xcpu_plugin.platform import McpuPlatform

    McpuPlatform.import_ir_kernels()

    assert ir.ops.rms_norm.impls["torch_xcpu"].inplace is False
    assert ir.ops.fused_add_rms_norm.impls["torch_xcpu"].inplace is False
    assert ir.ops.fused_add_rms_norm.impls["torch_xcpu_inplace"].inplace is True


def test_rms_norm_uses_vllm_ir(monkeypatch: pytest.MonkeyPatch, default_vllm_config):
    load_general_plugins()
    from vllm.model_executor.layers.layernorm import RMSNorm

    layer = RMSNorm(1024)
    assert layer.__class__ is RMSNorm


def test_ir_priority_uses_functional_kernel_for_compile():
    from types import SimpleNamespace

    from vllm.config import CompilationMode

    from vllm_xcpu_plugin.platform import McpuPlatform

    McpuPlatform.import_ir_kernels()
    eager_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    compile_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.DYNAMO_TRACE_ONCE)
    )

    eager_priority = McpuPlatform.get_default_ir_op_priority(eager_config)
    assert eager_priority.rms_norm == ["torch_xcpu", "native"]
    assert eager_priority.fused_add_rms_norm == [
        "torch_xcpu_inplace",
        "torch_xcpu",
        "native",
    ]
    compile_priority = McpuPlatform.get_default_ir_op_priority(compile_config)
    assert compile_priority.rms_norm == ["torch_xcpu", "native"]
    assert compile_priority.fused_add_rms_norm == [
        "torch_xcpu",
        "native",
    ]


def test_compile_config_disables_inductor_fusions():
    from types import SimpleNamespace

    from vllm.config import CompilationMode

    from vllm_xcpu_plugin.platform import McpuPlatform

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=[],
            backend="",
            inductor_compile_config={},
        ),
        parallel_config=SimpleNamespace(worker_cls=None),
        cache_config=SimpleNamespace(
            user_specified_block_size=True,
            block_size=256,
        ),
        model_config=None,
    )

    McpuPlatform.check_and_update_config(config)

    compile_config = config.compilation_config.inductor_compile_config
    assert compile_config["epilogue_fusion"] is False
    assert compile_config["pattern_matcher"] is False
    assert compile_config["combo_kernels"] is False
