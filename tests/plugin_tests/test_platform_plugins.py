# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
