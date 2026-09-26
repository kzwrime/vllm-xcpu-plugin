import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_xcpu_plugin.layers.fused_moe.moe_runner import XcpuMoERunner


@pytest.mark.parametrize(
    "case", ["fused", "fp16", "no_shared", "transform", "strided", "unit_scale"]
)
def test_scaled_shared_combine_preserves_runner_semantics(case):
    runner = object.__new__(XcpuMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_scaling_factor = 1.0 if case == "unit_scale" else 2.5
    runner.routed_output_transform = (lambda x: x + 1) if case == "transform" else None
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    shared = torch.randn(6, 6144, device="cpu", dtype=dtype).to("mcpu")
    routed = torch.randn_like(shared)
    if case == "no_shared":
        shared = None
    elif case == "strided":
        shared, routed = shared[:, ::2], routed[:, ::2]
    expected = MoERunner._combine_shared_expert_output(
        runner, None if shared is None else shared.clone(), routed.clone()
    ).cpu()
    with torch.profiler.profile() as profile:
        actual = runner._combine_shared_expert_output(shared, routed).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    names = {event.key for event in profile.key_averages()}
    assert ("torch_xcpu::moe_scaled_add_out" in names) == (case == "fused")
    if case == "fused":
        assert not {"aten::mul_", "aten::add"} & names


def test_scaled_shared_combine_aot_reload(monkeypatch, tmp_path):
    from pathlib import Path

    import torch_xcpu
    from torch._inductor.utils import run_and_get_code

    package = Path(torch_xcpu.__file__).parent
    library = next(package.glob("_C*.so"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv(
        "AOTI_EXTRA_CFLAGS", f"-include {package / 'include/aoti_torch_xcpu.h'}"
    )
    monkeypatch.setenv("AOTI_EXTRA_LDFLAGS", f"-Wl,-rpath,{package} {library}")
    monkeypatch.setenv("TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES", "torch_xcpu")
    runner = object.__new__(XcpuMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_scaling_factor = 2.5
    runner.routed_output_transform = None
    shared = torch.randn(6, 6144, device="cpu").bfloat16().to("mcpu")
    routed = torch.randn_like(shared)
    for tensor in (shared, routed):
        torch._dynamo.mark_dynamic(tensor, 0)
    target = XcpuMoERunner._combine_shared_expert_output
    torch._dynamo.reset()
    with (
        torch._dynamo.config.patch(enable_aot_compile=True),
        torch._inductor.config.patch(
            cpp_wrapper=True, enable_auto_functionalized_v2=False
        ),
    ):
        wrapper = torch.compile(target, fullgraph=True, dynamic=False)
        compiled, code = run_and_get_code(
            wrapper.aot_compile, ((runner, shared, routed), {})
        )
        code = "\n".join(code)
        (tmp_path / "scaled_add_generated_code.txt").write_text(code)
        assert (
            "aoti_torch_mcpu_moe_scaled_add_out" in code
            or 'aoti_torch_call_dispatcher("torch_xcpu::moe_scaled_add_out"' in code
        )
        assert "PyObject_CallObject" not in code
        artifact = tmp_path / "scaled_add.aot"
        compiled.save_compiled_function(str(artifact))
        with artifact.open("rb") as handle:
            loaded = torch.compiler.load_compiled_function(
                handle, f_globals=target.__globals__
            )
        for fn in (compiled, loaded):
            fn.disable_guard_check()
            for rows in (1, 6, 17):
                s = torch.randn(rows, 6144, device="cpu").bfloat16().to("mcpu")
                r = torch.randn_like(s)
                expected = s.cpu() + r.cpu() * 2.5
                torch.testing.assert_close(
                    fn(runner, s, r).cpu(), expected, rtol=0, atol=0
                )
    torch._dynamo.reset()
