import pytest
import torch
import torch_xcpu
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

from vllm_xcpu_plugin.layers.fused_moe.gate_linear import XcpuGateLinear


@pytest.fixture
def make_gate(monkeypatch):
    # GateLinear is replicated: constructing it needs TP metadata but no group
    # collective, so this unit test does not need to launch MPI.
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)

    def create(*, out_dtype=torch.float32, bias=False, force_fp32_compute=False):
        with torch.device("cpu"):
            gate = GateLinear(
                2,
                2,
                bias=bias,
                out_dtype=out_dtype,
                params_dtype=torch.bfloat16,
                force_fp32_compute=force_fp32_compute,
            )
        assert isinstance(gate, XcpuGateLinear)
        weight = torch.tensor(
            [[1, 2**-12], [1, 0]], dtype=gate.weight.dtype, device="cpu"
        )
        with torch.no_grad():
            gate.weight.copy_(weight)
            if bias:
                gate.bias.fill_(0.125)
        return gate.to("mcpu"), weight

    return create


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("late_dtype", [False, True])
def test_router_retains_fp32_logits(make_gate, compile, late_dtype):
    gate, weight = make_gate(out_dtype=None if late_dtype else torch.float32)
    if late_dtype:
        gate.set_out_dtype(torch.float32)
    x = torch.ones((1, 2), dtype=torch.bfloat16, device="cpu")
    expected = x.float() @ weight.float().T
    forward = torch.compile(gate, fullgraph=True) if compile else gate
    with torch.inference_mode():
        output, bias = forward(x.to("mcpu"))
        actual = output.cpu()
    assert bias is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 0] > actual[0, 1]


@pytest.mark.parametrize(
    "out_dtype,bias,force_fp32_compute",
    [
        (None, False, False),
        (torch.bfloat16, False, False),
        (torch.float32, True, False),
        (torch.float32, False, True),
    ],
)
def test_router_fallback_semantics(make_gate, out_dtype, bias, force_fp32_compute):
    gate, weight = make_gate(
        out_dtype=out_dtype, bias=bias, force_fp32_compute=force_fp32_compute
    )
    x = torch.ones((1, 2), dtype=torch.bfloat16, device="cpu")
    addend = torch.full((2,), 0.125, dtype=weight.dtype, device="cpu") if bias else None
    expected = torch.nn.functional.linear(x.to(weight.dtype), weight, addend)
    if out_dtype is not None:
        expected = expected.to(out_dtype)
    with torch.inference_mode():
        output, output_bias = gate(x.to("mcpu"))
        actual = output.cpu()
    assert output_bias is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_router_grouped_topk_uses_fp32_logits_without_cast(make_gate):
    gate, weight = make_gate()
    x_cpu = torch.ones((1, 2), dtype=torch.bfloat16, device="cpu")
    correction_cpu = torch.tensor([0.0, 2e-5], dtype=torch.float32, device="cpu")
    scores = (x_cpu.float() @ weight.float().T).sigmoid()
    expected_id = (scores + correction_cpu).argmax(-1)
    # Rounding logits to BF16 would lose the score margin and choose expert 1.
    rounded_id = ((x_cpu @ weight.T).float().sigmoid() + correction_cpu).argmax(-1)
    assert expected_id.item() == 0 and rounded_id.item() == 1
    x, correction = x_cpu.to("mcpu"), correction_cpu.to("mcpu")
    weights = torch.empty((1, 1), dtype=torch.float32, device="mcpu")
    ids = torch.empty((1, 1), dtype=torch.int32, device="mcpu")
    torch.mcpu.synchronize()
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile,
    ):
        logits, _ = gate(x)
        torch_xcpu.ops.grouped_topk(
            logits, 1, 1, 1, False, 1.0, correction, 1, weights, ids
        )
        torch.mcpu.synchronize()
    actual_ids, actual_weights = ids.cpu(), weights.cpu()
    torch.testing.assert_close(actual_ids[:, 0].long(), expected_id)
    torch.testing.assert_close(actual_weights, scores[:, :1])
    names = {event.key for event in profile.key_averages()}
    assert not {"aten::to", "aten::_to_copy", "prims::convert_element_type"} & names
