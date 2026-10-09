"""Direct EP binding and preservation of upstream shared-expert execution."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from torch_xcpu import ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

from vllm_xcpu_plugin.layers.fused_moe.ep_experts import (
    XcpuEPExperts,
    XcpuEPKernelAdapter,
)
from vllm_xcpu_plugin.layers.fused_moe.moe_runner import XcpuMoERunner


@pytest.mark.parametrize("version", [5, 6])
def test_ep_entry_rejects_incompatible_placement_and_inputs_before_dispatch(
    version,
):
    """Invalid EP inputs must not enqueue even the first collective."""
    plan = ops.initialize_fused_moe(
        torch.zeros(2, 256, 128, device="mcpu", dtype=torch.bfloat16),
        torch.zeros(2, 128, 128, device="mcpu", dtype=torch.bfloat16),
        backend=ops.MoeGroupedGemmBackend.PORTABLE,
    )
    metadata = torch.tensor([2, 0], dtype=torch.int64, device="cpu")
    # No MPI communicator is needed: none of the rejected calls may use it.
    communicator = torch.tensor([0], dtype=torch.int64, device="cpu")
    with pytest.raises(ValueError, match="linear expert map"):
        XcpuEPExperts(
            plan,
            metadata,
            communicator,
            torch.tensor([-1, -1, 0, 1], device="mcpu", dtype=torch.int32),
            version=version,
            ep_size=2,
            ep_rank=0,
            max_num_tokens=4,
            topk=6,
        )
    entry = XcpuEPExperts(
        plan,
        metadata,
        communicator,
        torch.tensor([0, 1, -1, -1], device="mcpu", dtype=torch.int32),
        version=version,
        ep_size=2,
        ep_rank=0,
        max_num_tokens=4,
        topk=6,
    )

    assert entry._op._schema.name == (
        f"torch_xcpu::fused_ep_moe_v{version}_PortableBf16MoeGroupedGemm"
    )
    assert "transport_version" not in [a.name for a in entry._op._schema.arguments]

    def unexpected_dispatch(*args):
        pytest.fail("invalid input reached the EP collective")

    entry._op = unexpected_dispatch
    ids = torch.zeros(5, 6, device="mcpu", dtype=torch.int32)
    weights = torch.ones(5, 6, device="mcpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        entry.forward(
            torch.empty(5, 128, device="mcpu", dtype=torch.bfloat16), weights, ids
        )
    with pytest.raises(ValueError, match="activation dtype"):
        entry.forward(
            torch.empty(4, 128, device="mcpu", dtype=torch.float32),
            weights[:4],
            ids[:4],
        )
    with pytest.raises(RuntimeError):
        entry.forward(
            torch.empty(4, 128, device="mcpu", dtype=torch.bfloat16),
            weights[:4, :5],
            ids[:4, :5],
        )


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_upstream_runner_owns_shared_and_merge(
    monkeypatch, sequence_parallel, latent, shared
):
    events = []

    class SharedLayer(torch.nn.Module):
        def forward(self, x):
            events.append("shared")
            return (2 * x + 1) * torch.sigmoid(x[:, :1])

    class RoutedEntry(torch.nn.Module):
        global_num_experts = 4

        def forward(self, x, weights, ids):
            events.append("routed")
            return 3 * x

    adapter = object.__new__(XcpuEPKernelAdapter)
    adapter.impl = object.__new__(mk.FusedMoEKernelModularImpl)
    adapter.impl.prepare_finalize = SimpleNamespace(output_is_reduced=lambda: True)
    adapter.ep_experts = RoutedEntry()
    runner = object.__new__(XcpuMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(
        hidden_dim=4,
        tp_size=2,
        ep_size=4,
        dp_size=2,
        pcp_size=1,
        is_sequence_parallel=sequence_parallel,
        skip_final_all_reduce=False,
        moe_parallel_config=SimpleNamespace(
            use_all2all_kernels=True,
            enable_eplb=False,
            use_fi_nvl_two_sided_kernels=False,
        ),
    )
    runner._shared_experts = (
        SharedExperts(
            SharedLayer(),
            runner.moe_config,
            False,
            lambda: adapter.can_overlap_shared_experts,
        )
        if shared
        else None
    )

    def routed_forward(x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return adapter.apply(
            x,
            None,
            None,
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            4,
            None,
            False,
            shared_experts,
            shared_experts_input,
        )

    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=adapter,
            supports_internal_mk=True,
            is_monolithic=False,
            skip_forward_padding=False,
            has_unpadded_output=False,
            topk_indices_dtype=torch.int32,
        ),
        _ensure_moe_quant_config_init=lambda: None,
        forward_modular=routed_forward,
    )

    def select_experts(hidden_states, **kwargs):
        events.append("route")
        m = hidden_states.shape[0]
        return torch.ones(m, 1), torch.zeros(m, 1, dtype=torch.int32)

    runner.router = SimpleNamespace(select_experts=select_experts)
    runner.routed_scaling_factor = 2.5
    runner.routed_input_transform = (lambda x: (x[:, :2], None)) if latent else None
    runner.routed_output_transform = (
        (lambda x: (torch.cat([x, x], -1), None)) if latent else None
    )
    runner.gate = None
    runner._sequence_parallel_context = nullcontext
    runner._encode_layer_name = lambda: "test"
    runner._forward_entry = lambda x, logits, sx, ids, *args: runner._forward_impl(
        x, logits, sx, ids
    )

    def reduce_shared(x):
        events.append("reduce")
        return x * 2

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner.tensor_model_parallel_all_reduce",
        reduce_shared,
    )
    for step in range(2):
        events.clear()
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10 + step
        result = runner.forward(x, x)
        routed_x = torch.cat([x[:, :2], x[:, :2]], -1) if latent else x
        expected = 7.5 * routed_x
        expected_events = ["route", "routed"]
        if shared:
            contribution = (2 * x + 1) * torch.sigmoid(x[:, :1])
            expected += contribution if sequence_parallel else contribution * 2
            expected_events.insert(0, "shared")
            if not sequence_parallel:
                expected_events.append("reduce")
            assert runner._shared_experts._output == [None, None]
        torch.testing.assert_close(result, expected)
        assert events == expected_events


@pytest.mark.parametrize("version", [5, 6])
def test_native_workspace_rejects_invalid_storage_before_dispatch(version):
    plan = ops.initialize_fused_moe(
        torch.zeros(2, 256, 128, device="mcpu", dtype=torch.bfloat16),
        torch.zeros(2, 128, 128, device="mcpu", dtype=torch.bfloat16),
        backend=ops.MoeGroupedGemmBackend.PORTABLE,
        allocate_scratch=False,
    )
    entry = XcpuEPExperts(
        plan,
        torch.tensor([2, 0], dtype=torch.int64, device="cpu"),
        torch.tensor([0], dtype=torch.int64, device="cpu"),
        torch.tensor([0, 1, -1, -1], device="mcpu", dtype=torch.int32),
        version=version,
        ep_size=2,
        ep_rank=0,
        max_num_tokens=4,
        topk=6,
    )
    native = entry._op
    captured = []
    entry._op = lambda *args: captured.extend(args)
    entry.forward(
        torch.empty(1, 128, device="mcpu", dtype=torch.bfloat16),
        torch.ones(1, 6, device="mcpu", dtype=torch.float32),
        torch.zeros(1, 6, device="mcpu", dtype=torch.int32),
    )
    assert len(captured) == 18
    assert [
        a.name
        for a in native._schema.arguments
        if a.alias_info and a.alias_info.is_write
    ] == ["output", "workspace"]
    workspace = captured[-1]
    assert workspace.numel() == entry.workspace_bytes
    assert entry.workspace_bytes % 64 == 0
    with pytest.raises(RuntimeError, match="workspace too small"):
        native(*captured[:-1], workspace[:-1])
    with pytest.raises(RuntimeError, match="aligned 1D uint8"):
        native(*captured[:-1], workspace[1:])
    with pytest.raises(RuntimeError, match="aligned 1D uint8"):
        native(*captured[:-1], workspace.view(torch.int8))
    alias_output = workspace[:256].view(torch.bfloat16).view(1, 128)
    with pytest.raises(RuntimeError, match="must not alias"):
        native(alias_output, *captured[1:])
    with pytest.raises(RuntimeError, match="must not alias"):
        native(captured[0], alias_output, *captured[2:])


@pytest.mark.parametrize("version", [5, 6])
def test_workspace_query_checks_capacity_and_overflow(version):
    query = getattr(
        torch.ops.torch_xcpu,
        f"fused_ep_moe_v{version}_workspace_size_PortableBf16MoeGroupedGemm",
    )
    args = [4, 17, 8, 128, 128, 4, 2, 0, 0, False, False]
    assert query(*args) > 0
    for index, value in ((0, 129), (1, 0), (2, 7), (3, -1), (4, 2**62), (6, 1)):
        invalid = args.copy()
        invalid[index] = value
        with pytest.raises(RuntimeError, match="invalid workspace dimensions"):
            query(*invalid)
    for bound in (2**62, 2**30, 2**22):
        invalid = args.copy()
        invalid[1] = bound
        with pytest.raises(RuntimeError, match="overflow|exceeds"):
            query(*invalid)


@pytest.mark.parametrize("version", [5, 6])
def test_workspace_rechecks_amx_scratch_after_thread_count_change(version):
    if not ops.moe_grouped_gemm_backend_available(
        ops.MoeWeightFormat.FP8_E4M3, ops.MoeGroupedGemmBackend.INTEL_AMX
    ):
        pytest.skip("AMX FP8 unavailable")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        plan = ops.initialize_fused_moe(
            torch.zeros(2, 256, 128, device="cpu", dtype=torch.float8_e4m3fn).to(
                "mcpu"
            ),
            torch.zeros(2, 128, 128, device="cpu", dtype=torch.float8_e4m3fn).to(
                "mcpu"
            ),
            torch.ones(2, 2, 1, device="mcpu", dtype=torch.float32),
            torch.ones(2, 1, 1, device="mcpu", dtype=torch.float32),
            scale_block_size=(128, 128),
            backend=ops.MoeGroupedGemmBackend.INTEL_AMX,
            allocate_scratch=False,
        )
        assert plan.params.gemm1.params.scratch.numel() == 0
        assert plan.params.gemm2.params.scratch.numel() == 0
        entry = XcpuEPExperts(
            plan,
            torch.tensor([2, 0], dtype=torch.int64, device="cpu"),
            torch.tensor([0], dtype=torch.int64, device="cpu"),
            torch.tensor([0, 1, -1, -1], device="mcpu", dtype=torch.int32),
            version=version,
            ep_size=2,
            ep_rank=0,
            max_num_tokens=1,
            topk=6,
        )
        # The communicator is deliberately invalid: capacity rejection must
        # happen on the submitter before any communication is enqueued.
        torch.set_num_threads(8)
        with pytest.raises(RuntimeError, match="workspace too small"):
            entry(
                torch.empty(1, 128, device="mcpu", dtype=torch.bfloat16),
                torch.ones(1, 6, device="mcpu", dtype=torch.float32),
                torch.zeros(1, 6, device="mcpu", dtype=torch.int32),
            )
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("version", [5, 6])
@pytest.mark.parametrize(
    "local_experts,arena_gib", [(1, 3), (2, 3.75), (4, 5.25), (8, 10)]
)
def test_compact_workspace_ep64_capacity(version, local_experts, arena_gib):
    """Production-size query only: never allocate these multi-GiB buffers.

    With four experts the unpermute stage, not either GEMM, sets the peak.
    Original top-k routing/sort metadata remains full-sized in all cases.
    """
    query = getattr(
        torch.ops.torch_xcpu,
        f"fused_ep_moe_v{version}_workspace_size_AccFp8MoeGroupedGemm",
    )
    metadata_bytes = 8_422_272
    assert query(64, 1024, 8, 6144, 2048, local_experts, 2, 128, 128, False, False) == (
        int(arena_gib * 2**30) + metadata_bytes
    )
