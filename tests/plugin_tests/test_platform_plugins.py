# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_norm_split_views_eager_and_compile(dtype, monkeypatch):
    """SP split views must select the fused provider without input copies."""
    from pathlib import Path

    import torch_mcpu  # noqa: F401
    import torch_xcpu
    from torch._inductor.utils import run_and_get_code
    from vllm import ir
    from vllm.ir.op import enable_torch_wrap

    from vllm_xcpu_plugin.platform import McpuPlatform

    McpuPlatform.import_ir_kernels()
    package = Path(torch_xcpu.__file__).parent
    library = next(package.glob("_C*.so"))
    monkeypatch.setenv(
        "AOTI_EXTRA_CFLAGS", f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    monkeypatch.setenv("AOTI_EXTRA_LDFLAGS", f"-Wl,-rpath,{package} {library}")
    monkeypatch.setenv("TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES", "torch_xcpu")

    def fn(packed, weight):
        x, residual = packed.split(64, dim=-1)
        return ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, 1e-6)

    weight = torch.ones(64, dtype=dtype, device="mcpu")
    op = ir.ops.fused_add_rms_norm
    torch._dynamo.reset()
    with enable_torch_wrap(False), op.set_priority(["torch_xcpu", "native"]):
        compiled = torch.compile(
            fn, fullgraph=True, dynamic=True, options={"cpp_wrapper": True}
        )
        for rows in (3, 7):
            cpu = torch.randn(rows, 128, dtype=dtype, device="cpu")
            packed = cpu.to("mcpu")
            x, residual = packed.split(64, dim=-1)
            assert op.dispatch(x, residual, weight, 1e-6).provider == "torch_xcpu"
            summed = cpu[:, :64].float() + cpu[:, 64:].float()
            expected = (
                summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            (out, residual_out), code = run_and_get_code(compiled, packed, weight)
            torch.testing.assert_close(out.cpu(), expected, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(
                residual_out.cpu(), summed.to(dtype), atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(packed.cpu(), cpu)
            if rows == 3:
                code = "\n".join(code)
                assert "aoti_torch_mcpu_fused_add_rms_norm_out" in code
                assert "torch.ops.aten.mean.dim" not in code
                assert "aoti_torch_mcpu_pow_Tensor_Scalar" not in code
                assert "aten::clone" not in code

    with (
        enable_torch_wrap(False),
        op.set_priority(["torch_xcpu_inplace", "torch_xcpu", "native"]),
    ):
        assert op.dispatch(x, residual, weight, 1e-6).provider == "torch_xcpu_inplace"
        out, residual_out = fn(packed, weight)
        assert out.data_ptr() == x.data_ptr()
        assert residual_out.data_ptr() == residual.data_ptr()
        torch.testing.assert_close(out.cpu(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            residual_out.cpu(), summed.to(dtype), atol=2e-2, rtol=2e-2
        )
    torch._dynamo.reset()
