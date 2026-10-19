"""Direct EP binding and preservation of upstream shared-expert execution."""

import math
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
    XcpuSharedExpertWeights,
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
    with pytest.raises(ValueError, match="expert map"):
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
    assert len(captured) == 19
    assert [
        a.name
        for a in native._schema.arguments
        if a.alias_info and a.alias_info.is_write
    ] == ["output", "workspace"]
    workspace = captured[-2]
    assert workspace.numel() == entry.workspace_bytes
    assert entry.workspace_bytes % 64 == 0
    with pytest.raises(RuntimeError, match="workspace too small"):
        native(*captured[:-2], workspace[:-1], captured[-1])
    with pytest.raises(RuntimeError, match="aligned 1D uint8"):
        native(*captured[:-2], workspace[1:], captured[-1])
    with pytest.raises(RuntimeError, match="aligned 1D uint8"):
        native(*captured[:-2], workspace.view(torch.int8), captured[-1])
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


def _align128(nbytes):
    return (nbytes + 127) // 128 * 128


def v5_region_bytes(chunk, topk, hidden, local_experts):
    """One rank's fixed dispatch region: header, then C hidden rows, C*K
    weights, C*K route refs and the static [E, C] route index, every section
    128-byte aligned, padded to lcm(B, 128)."""
    row = hidden * 2
    full = (
        _align128((2 + local_experts) * 4)
        + _align128(chunk * row)
        + 2 * _align128(chunk * topk * 4)
        + chunk * local_experts * 4
    )
    unit = math.lcm(row, 128)
    return -(-full // unit) * unit


def v5_metadata_bytes(peers, rows, topk, local_experts, chunk=None):
    """V5 prefix: return rows, the [E, P*C] GEMM1 route index, expert
    offsets, the [P,E] route bases (the ACC indexed GEMM1 has no scratch) and
    the internal 128-byte base shift."""
    index = local_experts * peers * (chunk or rows)
    return (
        _align128(rows * topk * 4)
        + _align128(index * 4)
        + _align128((local_experts + 1) * 4)
        + _align128(peers * local_experts * 4)
        + 128
    )


def v5_arena_bytes(peers, rows, topk, local_experts, hidden, intermediate):
    """Unchunked V5 arena: activated (+ GEMM2 scratch, 0 for ACC) or the
    expert output in S/B-row window-region slots + FP32 reduce, followed by
    the GEMM2 output. Both [P,S] windows live outside the workspace."""
    capacity = peers * rows
    routes = capacity * min(topk, local_experts)
    activated = routes * intermediate * 2
    expert_and_reduce = (
        peers * v5_region_bytes(rows, topk, hidden, local_experts) + rows * hidden * 4
    )
    return max(activated, expert_and_reduce) + routes * hidden * 2


@pytest.mark.parametrize("version", [5, 6])
@pytest.mark.parametrize(
    "local_experts,arena_gib", [(1, 1.5234375), (2, 2.5), (4, 5), (8, 10)]
)
def test_compact_workspace_ep64_capacity(version, local_experts, arena_gib):
    """Production-size query only: never allocate these multi-GiB buffers.

    With four experts GEMM1 sets the peak after per-worker FP32 reduction.
    V5 omits unpacked IDs, the forward permutation and the valid-row tensor,
    and adds the pack block counters.
    """
    query = getattr(
        torch.ops.torch_xcpu,
        f"fused_ep_moe_v{version}_workspace_size_AccFp8MoeGroupedGemm",
    )
    size = query(64, 1024, 8, 6144, 2048, local_experts, 2, 128, 128, False, False)
    if version == 5:
        # GEMM1 reads the window through the indexed backend: no permuted
        # input, GEMM1 output or GEMM1 scratch in the arena.
        assert size == (
            v5_arena_bytes(64, 1024, 8, local_experts, 6144, 2048)
            + v5_metadata_bytes(64, 1024, 8, local_experts)
        )
    else:
        assert size == int(arena_gib * 2**30) + 8_422_272


@pytest.mark.parametrize("version", [5, 6])
def test_glm52_acc_mxfp4_workspace_after_row_reduce(version):
    query = getattr(
        torch.ops.torch_xcpu,
        f"fused_ep_moe_v{version}_workspace_size_AccMxfp4MoeGroupedGemm",
    )
    # TP4/SP4: M=2048/4. Quota 1024 is clamped to M, so no chunking.
    size = query(64, 512, 8, 6144, 2048, 4, 2, 1, 32, False, False, 1024)
    if version == 5:
        assert size == v5_arena_bytes(64, 512, 8, 4, 6144, 2048) + v5_metadata_bytes(
            64, 512, 8, 4
        )
    else:
        assert size == 2560 * 2**20 + 4_211_584
        assert 2_822_783_872 - size == 128 * 2**20


@pytest.mark.parametrize("version", [5, 6])
def test_chunk_workspace_bounds_and_disabled_compatibility(version):
    query = getattr(
        torch.ops.torch_xcpu,
        f"fused_ep_moe_v{version}_workspace_size_AccFp8MoeGroupedGemm",
    )
    args = (64, 6144, 8, 6144, 2048, 4, 2, 128, 128, False, False)
    # Only a size query: no multi-GiB allocation. Unchunked dispatch would
    # exceed MPI's int32 displacement; the bounded layout must remain usable.
    chunked = query(*args, 1024)
    base = query(64, 1024, *args[2:])
    # Full-input return indices and the FP32 [M,H] accumulator.
    extra = (6144 - 1024) * 8 * 4 + 6144 * 6144 * 4
    if version == 6:
        # Persistent dense send: it cannot alias compute while later chunks
        # still need it. V5 packs every round into the [P,S] arena instead.
        packed_bytes = 6144 * 8 * (8 + 8 * 8 + 6144 * 2) + 8
        extra += (packed_bytes + 63) // 64 * 64
    assert chunked == base + extra
    with pytest.raises(RuntimeError, match="receive window exceeds"):
        query(128, 6144, *args[2:], 2048)
    if version == 6:
        # Persistent send and bounded receive have independent MPI int32 limits.
        with pytest.raises(RuntimeError, match="dispatch capacity exceeds"):
            query(64, 32768, *args[2:], 1024)
    else:
        # V5 send and receive are both [P,S]: only C bounds them. Only the
        # [M,K] return indices and the FP32 [M,H] accumulator grow with M.
        growth = (32768 - 6144) * (8 * 4 + 6144 * 4)
        assert query(64, 32768, *args[2:], 1024) == chunked + growth
    small = (4, 17, 8, 128, 128, 4, 2, 128, 128, False, False)
    assert query(*small) == query(*small, 0) == query(*small, 17) == query(*small, 1024)
    assert query(*small, 3) < query(*small)
    with pytest.raises(RuntimeError, match="nonnegative"):
        query(*small, -1)


@pytest.mark.parametrize("version", [5, 6])
def test_post_sp_input_bound_does_not_scale_receive_quota(monkeypatch, version):
    monkeypatch.setenv("VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK", "8")
    plan = ops.initialize_fused_moe(
        torch.zeros(2, 256, 128, device="mcpu", dtype=torch.bfloat16),
        torch.zeros(2, 128, 128, device="mcpu", dtype=torch.bfloat16),
        backend=ops.MoeGroupedGemmBackend.PORTABLE,
        allocate_scratch=False,
    )
    # ceil(33/2)=17 supplied by vLLM's SP prepare/finalize contract.
    entry = XcpuEPExperts(
        plan,
        torch.tensor([2, 0]),
        torch.tensor([0]),
        torch.tensor([0, 1, -1, -1], device="mcpu", dtype=torch.int32),
        version=version,
        ep_size=2,
        ep_rank=0,
        max_num_tokens=17,
        topk=6,
    )
    captured = []
    entry._op = lambda *args: captured.extend(args)
    entry(
        torch.empty(17, 128, device="mcpu", dtype=torch.bfloat16),
        torch.ones(17, 6, device="mcpu"),
        torch.zeros(17, 6, device="mcpu", dtype=torch.int32),
    )
    assert captured[14] == 17
    assert captured[-1] == 8
    monkeypatch.setenv("VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        XcpuEPExperts(
            plan,
            torch.tensor([2, 0]),
            torch.tensor([0]),
            torch.tensor([0, 1, -1, -1], device="mcpu", dtype=torch.int32),
            version=version,
            ep_size=2,
            ep_rank=0,
            max_num_tokens=17,
            topk=6,
        )


@pytest.mark.parametrize(
    "mapping,accepted",
    [
        ([1, 0, -1, -1], True),
        ([-1, 0, -1, -1], True),
        ([-1, -1, -1, -1], True),
        ([0, 0, -1, -1], False),
        ([0, 2, -1, -1], False),
        ([0, -2, -1, -1], False),
        ([0, 1, 0, -1], False),
    ],
)
@pytest.mark.parametrize(
    "backend", [ops.MoeGroupedGemmBackend.PORTABLE_LUT, ops.MoeGroupedGemmBackend.ACC]
)
def test_v5_checks_map_only_at_initialization(mapping, accepted, backend):
    plan = ops.initialize_fused_moe(
        torch.zeros(2, 256, 128, device="cpu", dtype=torch.float8_e4m3fn).to("mcpu"),
        torch.zeros(2, 128, 128, device="cpu", dtype=torch.float8_e4m3fn).to("mcpu"),
        torch.ones(2, 2, 1, device="mcpu", dtype=torch.float32),
        torch.ones(2, 1, 1, device="mcpu", dtype=torch.float32),
        scale_block_size=(128, 128),
        backend=backend,
        allocate_scratch=False,
    )

    def initialize():
        return XcpuEPExperts(
            plan,
            torch.tensor([2, 0], dtype=torch.int64, device="cpu"),
            torch.tensor([0], dtype=torch.int64, device="cpu"),
            torch.tensor(mapping, dtype=torch.int32, device="mcpu"),
            version=5,
            ep_size=2,
            ep_rank=0,
            max_num_tokens=4,
            topk=8,
        )

    if accepted:
        entry = initialize()
        assert plan.backend_type.__name__ in entry._op._schema.name
    else:
        with pytest.raises(ValueError, match="injective local expert map"):
            initialize()


# -----------------------------------------------------------------------------
# Fused shared experts (VLLM_XCPU_FUSE_SHARED_EXPERTS): the v5 op takes the
# shared MLP's packed linear weights and the routed scaling factor.
# -----------------------------------------------------------------------------
def _v5_entry(version=5):
    plan = ops.initialize_fused_moe(
        torch.zeros(2, 256, 128, device="mcpu", dtype=torch.bfloat16),
        torch.zeros(2, 128, 128, device="mcpu", dtype=torch.bfloat16),
        backend=ops.MoeGroupedGemmBackend.PORTABLE,
    )
    return XcpuEPExperts(
        plan,
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, 1, -1, -1], device="mcpu", dtype=torch.int32),
        version=version,
        ep_size=2,
        ep_rank=0,
        max_num_tokens=4,
        topk=6,
    )


def _unquantized_shared(intermediate=64, gate=True):
    return XcpuSharedExpertWeights(
        w13=torch.zeros(2 * intermediate, 128, device="mcpu", dtype=torch.bfloat16),
        w2=torch.zeros(128, intermediate, device="mcpu", dtype=torch.bfloat16),
        w13_scale=None,
        w2_scale=None,
        linear_backend=-1,
        scale_block_size=(0, 0),
        gate=torch.zeros(1, 128, device="mcpu", dtype=torch.bfloat16) if gate else None,
    )


def test_bind_shared_experts_passes_linear_weights_and_scale():
    entry = _v5_entry()
    routed_only = entry.workspace_bytes
    shared = _unquantized_shared()
    entry.bind_shared_experts(shared, 2.5)
    assert entry.workspace_bytes == entry._workspace_size(64) > routed_only
    calls = []
    entry._op = lambda *args: calls.append(args)
    entry.forward(
        torch.zeros(3, 128, device="mcpu", dtype=torch.bfloat16),
        torch.ones(3, 6, device="mcpu"),
        torch.zeros(3, 6, device="mcpu", dtype=torch.int32),
    )
    (args,) = calls
    assert args[19:23] == (shared.w13, shared.w2, None, None)
    assert args[23:] == (-1, 0, 0, shared.gate, 2.5)
    with pytest.raises(ValueError, match="only XCPU EP v5"):
        _v5_entry(6).bind_shared_experts(shared, 1.0)


def _fusable_runner(tp_size=1, sequence_parallel=True, gate=True):
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    def linear(out_features, in_features):
        return SimpleNamespace(
            tp_size=tp_size,
            bias=None,
            quant_method=object.__new__(UnquantizedLinearMethod),
            weight=torch.zeros(out_features, in_features, dtype=torch.bfloat16),
        )

    # The type name selects the gate_up -> SiLU -> down (-> gate) contract.
    mlp = type("Qwen2MoeMLP", (), {})()
    mlp.gate_up_proj, mlp.down_proj = linear(128, 128), linear(128, 64)
    mlp.act_fn = object.__new__(SiluAndMul)
    mlp.expert_gate = linear(1, 128) if gate else None
    if mlp.expert_gate is not None:
        mlp.expert_gate.tp_size = 1
    bound = []
    ep_experts = SimpleNamespace(
        version=5,
        activation_dtype=torch.bfloat16,
        compute=SimpleNamespace(params=SimpleNamespace(hidden=128)),
        bind_shared_experts=lambda shared, scale: bound.append((shared, scale)),
    )
    runner = object.__new__(XcpuMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(
        tp_size=tp_size, is_sequence_parallel=sequence_parallel
    )
    runner._shared_experts = SimpleNamespace(_layer=mlp)
    runner.routed_experts = SimpleNamespace(_xcpu_ep_experts=ep_experts)
    runner.routed_scaling_factor = 2.5
    runner.routed_input_transform = runner.routed_output_transform = None
    runner._forward_entry = None
    return runner, mlp, bound


def test_install_fused_shared_experts_detaches_the_runner_shared_path(monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import _moe_forward

    from vllm_xcpu_plugin.layers.fused_moe.shared_experts import (
        install_fused_shared_experts,
    )

    monkeypatch.setenv("VLLM_XCPU_FUSE_SHARED_EXPERTS", "0")
    runner, mlp, bound = _fusable_runner()
    assert install_fused_shared_experts(runner) == 0 and not bound

    # On by default.
    monkeypatch.delenv("VLLM_XCPU_FUSE_SHARED_EXPERTS")
    assert install_fused_shared_experts(runner) == 1
    ((shared, scale),) = bound
    assert shared.w13 is mlp.gate_up_proj.weight and shared.w2 is mlp.down_proj.weight
    assert shared.gate is mlp.expert_gate.weight and shared.linear_backend == -1
    assert scale == 2.5
    # The op returns the complete output: no runner-side shared or scaling.
    assert runner._shared_experts is None and runner.routed_scaling_factor == 1.0
    assert runner._forward_entry is _moe_forward
    routed = torch.randn(3, 4)
    assert runner._combine_shared_expert_output(None, routed.clone()).equal(routed)


@pytest.mark.parametrize(
    "tp_size,sequence_parallel", [(2, False), (2, True)], ids=["tp", "tp_linear"]
)
def test_install_fused_shared_experts_keeps_tensor_parallel_mlp(
    monkeypatch, tp_size, sequence_parallel
):
    from vllm_xcpu_plugin.layers.fused_moe.shared_experts import (
        install_fused_shared_experts,
    )

    monkeypatch.setenv("VLLM_XCPU_FUSE_SHARED_EXPERTS", "1")
    runner, _, bound = _fusable_runner(tp_size, sequence_parallel)
    assert install_fused_shared_experts(runner) == 0 and not bound
    assert runner._shared_experts is not None and runner.routed_scaling_factor == 2.5
