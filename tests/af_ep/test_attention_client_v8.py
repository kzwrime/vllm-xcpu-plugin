from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.af_ep.attn import runtime
from vllm_xcpu_plugin.af_ep.attn.client_v8 import ExpertsClientV8


def test_attention_runs_remote_layers_without_local_weights_or_double_reduce(
    monkeypatch, make_af_session_v8
):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts

    monkeypatch.setattr(
        moe_runner,
        "tensor_model_parallel_all_reduce",
        lambda _: pytest.fail("remote output reduced twice"),
    )
    client = ExpertsClientV8(make_af_session_v8())
    monkeypatch.setattr(runtime, "get_remote_experts_client", lambda: client)
    methods = {
        i: XcpuRoutedExperts._get_quant_method(
            None,
            f"model.layers.{i}.mlp.experts",
            None,
            SimpleNamespace(moe_parallel_config=SimpleNamespace(sp_size=1)),
        )
        for i in (1, 3)
    }
    client.initialize(16, 6, torch.bfloat16)

    def execute(
        output,
        hidden,
        ids,
        weights,
        experts,
        capacity,
        layer_idx,
        metadata,
        comm,
        workspace,
        *extra,
    ):
        del experts, capacity, metadata, comm, workspace, extra
        assert ids.dtype == torch.int32 and ids.is_contiguous()
        assert weights.dtype == torch.float32 and weights.is_contiguous()
        output.copy_(hidden * weights.sum(dim=1, keepdim=True) * layer_idx)

    client._op = execute
    outputs = []
    for layer_idx, rows in ((1, 2), (3, 5)):
        method = methods[layer_idx]
        layer = torch.nn.Module()
        layer.global_num_experts, layer.local_num_experts = 8, 4
        method.create_weights(layer, 8, 16, 8, torch.bfloat16)
        layer.quant_method = method
        assert (
            XcpuRoutedExperts.load_weights(
                layer, iter([("0.gate_proj.weight_packed", torch.ones(1))])
            )
            == ()
        )
        method.process_weights_after_loading(layer)
        assert not list(layer.parameters()) and not list(layer.buffers())
        output = method.apply(
            layer,
            torch.ones(rows, 16, dtype=torch.bfloat16),
            torch.ones(6, rows, dtype=torch.bfloat16).t(),
            torch.arange(6, dtype=torch.int64).expand(rows, 6),
            None,
            None,
        )
        runner = SimpleNamespace(
            _quant_method=method,
            moe_config=SimpleNamespace(
                is_sequence_parallel=False,
                skip_final_all_reduce=False,
                tp_size=2,
                ep_size=2,
            ),
        )
        runner._fused_output_is_reduced = (
            moe_runner.MoERunner._fused_output_is_reduced.fget(runner)
        )
        outputs.append(
            moe_runner.MoERunner._maybe_reduce_final_output(runner, output, None)
        )
    torch.testing.assert_close(outputs[0], torch.full((2, 16), 6, dtype=torch.bfloat16))
    torch.testing.assert_close(
        outputs[1], torch.full((5, 16), 18, dtype=torch.bfloat16)
    )


def test_attention_uses_remote_expert_partition_when_rank_counts_differ(
    make_af_session_v8,
):
    client = ExpertsClientV8(
        make_af_session_v8(num_attention_ranks=4, num_expert_ranks=1)
    )
    client.initialize(16, 6, torch.bfloat16)
    observed = []

    def execute(*args):
        observed.append((args[4], int(args[7][4])))
        args[0].zero_()

    client._op = execute
    client.execute_layer(
        layer_idx=1,
        hidden_states=torch.ones(2, 16, dtype=torch.bfloat16),
        topk_weights=torch.ones(2, 6),
        topk_ids=torch.zeros(2, 6, dtype=torch.int32),
        num_experts=8,
        num_local_experts=2,
    )
    # Native A entry derives E=global/F from the remote topology, not A's shard.
    assert observed == [(8, 1)]


@pytest.mark.parametrize(
    "rows,num_f,capacity", [(3, 2, 8), (128, 8, 256), (129, 8, 256)]
)
def test_attention_traces_one_workspace_without_exposing_aliased_views(
    make_af_session_v8, rows, num_f, capacity
):
    from torch.fx.experimental.proxy_tensor import make_fx

    client = ExpertsClientV8(
        make_af_session_v8(num_expert_ranks=num_f, max_rows=capacity)
    )
    client.initialize(16, 6, torch.bfloat16)

    def remote(x, weights, ids):
        return client.execute_layer(
            layer_idx=1,
            hidden_states=x,
            topk_weights=weights,
            topk_ids=ids,
            num_experts=8,
            num_local_experts=4,
        )

    graph = make_fx(remote, tracing_mode="fake", _allow_non_fake_inputs=True)(
        torch.ones(rows, 16, dtype=torch.bfloat16),
        torch.ones(rows, 6),
        torch.zeros(rows, 6, dtype=torch.int32),
    )
    graph.graph.eliminate_dead_code()
    endpoints = [
        n.target
        for n in graph.graph.nodes
        if n.op == "call_function" and "fused_af_a_dispatch_combine" in str(n.target)
    ]
    assert endpoints == [
        torch.ops.torch_xcpu.fused_af_a_dispatch_combine_v8_bf16.default
    ]


def test_attention_reuses_workspace_across_pack_threshold(make_af_session_v8):
    client = ExpertsClientV8(make_af_session_v8(num_expert_ranks=8, max_rows=256))
    client.initialize(16, 6, torch.bfloat16)
    buffers = []

    def execute(*args):
        buffers.append(args[9])
        args[0].zero_()

    client._op = execute
    for rows in (128, 129, 256, 1):
        client.execute_layer(
            layer_idx=1,
            hidden_states=torch.ones(rows, 16, dtype=torch.bfloat16),
            topk_weights=torch.ones(rows, 6),
            topk_ids=torch.zeros(rows, 6, dtype=torch.int32),
            num_experts=8,
            num_local_experts=4,
        )
    assert all(buffer is buffers[0] for buffer in buffers)
    align128 = lambda n: (n + 127) // 128 * 128
    # v5-aligned A-side workspace: return_rows + FP32 reduce + 128B shift
    # (dispatch send buffer lives in A's combine MPI window, not workspace).
    assert buffers[0].numel() == align128(256 * 6 * 4) + align128(256 * 16 * 4) + 128


def test_attention_pads_non_divisible_experts_and_accepts_sp_padding_and_topk3(
    make_af_session_v8,
):
    client = ExpertsClientV8(make_af_session_v8(num_expert_ranks=3, max_rows=16))
    client.register_num_experts(10)
    client.initialize(16, 3, torch.bfloat16)
    observed = []

    def execute(*args):
        observed.append((args[4], args[1].shape[0], args[2].shape))
        args[0].zero_()

    client._op = execute
    client.execute_layer(
        layer_idx=1,
        hidden_states=torch.ones(8, 16, dtype=torch.bfloat16),
        topk_weights=torch.ones(5, 3),
        topk_ids=torch.zeros(5, 3, dtype=torch.int32),
        num_experts=10,
        num_local_experts=4,
    )
    # 10 experts across 3 F ranks pads to ceil(10/3)*3 = 12.
    assert observed == [(12, 8, (5, 3))]


def test_install_fused_shared_experts_binds_af_v8_client(
    monkeypatch, make_af_session_v8, default_vllm_config
):
    del default_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_xcpu_plugin.af_ep.attn.remote_fused_moe import (
        RemoteExpertsFusedMoEMethod,
    )
    from vllm_xcpu_plugin.layers.fused_moe.shared_experts import (
        install_fused_shared_experts,
    )

    monkeypatch.setenv("VLLM_XCPU_FUSE_SHARED_EXPERTS", "1")
    client = ExpertsClientV8(make_af_session_v8(num_expert_ranks=2, max_rows=16))
    method = RemoteExpertsFusedMoEMethod(
        SimpleNamespace(
            moe_parallel_config=SimpleNamespace(sp_size=1),
            num_experts=8,
        ),
        layer_name="model.layers.2.mlp.experts",
        client=client,
    )
    assert method.moe_kernel.supports_num_valid_tokens() is True

    class DeepseekV2MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act_fn = SiluAndMul()
            self.gate_up_proj = torch.nn.Module()
            self.gate_up_proj.tp_size = 1
            self.gate_up_proj.quant_method = UnquantizedLinearMethod()
            self.gate_up_proj.weight = torch.nn.Parameter(
                torch.zeros(16, 16, dtype=torch.bfloat16)
            )
            self.down_proj = torch.nn.Module()
            self.down_proj.tp_size = 1
            self.down_proj.quant_method = UnquantizedLinearMethod()
            self.down_proj.weight = torch.nn.Parameter(
                torch.zeros(16, 8, dtype=torch.bfloat16)
            )

    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner._shared_experts = SimpleNamespace(_layer=DeepseekV2MLP())
    runner.routed_experts = SimpleNamespace(quant_method=method)
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 2.5
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=True,
        tp_size=2,
        hidden_dim=16,
        in_dtype=torch.bfloat16,
    )
    runner._select_forward = lambda: "routed_only"

    model = torch.nn.Module()
    model.runner = runner
    assert install_fused_shared_experts(model) == 1
    assert runner._shared_experts is None
    assert runner.routed_scaling_factor == 1.0
    assert 2 in client._shared_by_layer
    assert client._shared_by_layer[2].routed_scaling_factor == 2.5
    assert client._max_shared_intermediate == 8


def test_v8_initialization_validates_protocol(monkeypatch, make_af_session_v8):
    from torch_xcpu.ops_defs import moe_af_v8

    monkeypatch.setattr(moe_af_v8, "_ENABLE_CHECKS", True)
    session = make_af_session_v8()
    moe_af_v8.moe_af_v8_initialize_check(
        8, 16, 6, torch.bfloat16, session.metadata, session.communicator_handle
    )
    old_metadata = session.metadata.clone()
    old_metadata[0] = 1
    with pytest.raises(RuntimeError):
        moe_af_v8.moe_af_v8_initialize_check(
            8, 16, 6, torch.bfloat16, old_metadata, session.communicator_handle
        )


def test_attention_routed_experts_skips_ep_padding_for_unequal_af_topologies(
    monkeypatch, make_af_session_v8, default_vllm_config
):
    """A's DP*TP size must not pad num_experts before ExpertsClientV8 pads by F."""
    del default_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )

    from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts
    from vllm_xcpu_plugin.worker.worker_v1 import McpuWorker, Worker

    # 10 experts: not divisible by A=3 (DP3xTP1) and not divisible by F=4.
    session = make_af_session_v8(num_attention_ranks=3, num_expert_ranks=4, max_rows=8)
    client = ExpertsClientV8(session)
    monkeypatch.setattr(runtime, "get_remote_experts_client", lambda: client)

    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=3,
        ep_size=3,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="mpi_alltoallv_v8",
        enable_eplb=False,
    )
    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=8,
        top_k=6,
        global_num_experts=10,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
        num_fused_shared_experts=0,
        rocm_aiter_enabled=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=10,
        experts_per_token=6,
        hidden_dim=16,
        intermediate_size=8,
        num_local_experts=expert_map_manager.local_num_experts,
        num_logical_experts=10,
        activation=MoEActivation.SILU,
        device=torch.device("mcpu"),
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        max_num_tokens=8,
    )
    layer = XcpuRoutedExperts(
        layer_name="model.layers.0.mlp.experts",
        params_dtype=torch.bfloat16,
        moe_config=moe_config,
        quant_config=None,
        expert_map_manager=expert_map_manager,
    )
    assert layer._xcpu_padded_experts is False
    assert layer.global_num_experts == 10

    monkeypatch.setattr(Worker, "load_model", lambda self, **kw: None)
    monkeypatch.setattr(
        "vllm_xcpu_plugin.layers.fused_moe.shared_experts.install_fused_shared_experts",
        lambda model: 0,
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    worker = object.__new__(McpuWorker)
    worker.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            n_routed_experts=10, hidden_size=16, num_experts_per_tok=6
        ),
        dtype=torch.bfloat16,
    )
    worker.model_runner = SimpleNamespace(
        get_model=lambda: None, get_draft_model=lambda: None
    )
    worker._af_client = client
    worker._engram_client = None
    worker.load_model()

    assert session.local_experts == 3
    observed = []
    client._op = lambda *args: (observed.append(args[4]), args[0].zero_())
    layer.quant_method.apply(
        layer,
        torch.ones(2, 16, dtype=torch.bfloat16),
        torch.ones(2, 6),
        torch.zeros(2, 6, dtype=torch.int32),
        None,
        None,
    )
    assert observed == [12]


def test_v8_fake_registrations_reject_invalid_route_dtypes_and_missing_scales():
    from torch.fx.experimental.proxy_tensor import make_fx

    op = torch.ops.torch_xcpu.fused_af_a_dispatch_combine_v8_bf16.default
    out = torch.empty(2, 16, dtype=torch.bfloat16)
    x = torch.empty(2, 16, dtype=torch.bfloat16)
    bad_ids = torch.zeros(2, 6, dtype=torch.int64)
    weights = torch.ones(2, 6, dtype=torch.float32)
    meta = torch.zeros(8, dtype=torch.int64)
    comm = torch.zeros(1, dtype=torch.int64)
    ws = torch.empty(1024, dtype=torch.uint8)

    with pytest.raises(RuntimeError):
        make_fx(
            lambda o, h, i, w, m, c, s: op(o, h, i, w, 8, 4, 0, m, c, s),
            tracing_mode="fake",
        )(out, x, bad_ids, weights, meta, comm, ws)

    ids = torch.zeros(2, 6, dtype=torch.int32)
    q_w1 = torch.empty(16, 16, dtype=torch.float8_e4m3fn)
    q_w2 = torch.empty(16, 8, dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError):
        make_fx(
            lambda o, h, i, w, m, c, s, w1, w2: op(
                o, h, i, w, 8, 4, 0, m, c, s, 0, w1, w2, None, None, 0
            ),
            tracing_mode="fake",
        )(out, x, ids, weights, meta, comm, ws, q_w1, q_w2)
