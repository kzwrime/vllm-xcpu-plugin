from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)

import vllm_xcpu_plugin.layers.fused_moe.prepare_finalize_factory as pf_factory
from vllm_xcpu_plugin.layers.fused_moe import (
    torch_all_to_all_single_prepare_finalize as torch_a2a,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v2 import (
    MpiAlltoallvPrepareAndFinalizeV2,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v3 import (
    MpiAlltoallvPrepareAndFinalizeV3,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v4 import (
    MpiAlltoallvPrepareAndFinalizeV4,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v5 import (
    MpiAlltoallvPrepareAndFinalizeV5,
)
from vllm_xcpu_plugin.layers.fused_moe.torch_all_to_all_single_prepare_finalize import (
    TorchAlltoallSinglePrepareAndFinalize,
)


@pytest.mark.parametrize(
    "prepare_finalize_cls",
    [
        MpiAlltoallvPrepareAndFinalizeV2,
        MpiAlltoallvPrepareAndFinalizeV3,
        MpiAlltoallvPrepareAndFinalizeV4,
        MpiAlltoallvPrepareAndFinalizeV5,
    ],
)
def test_mpi_prepare_rejects_activation_quantization(prepare_finalize_cls):
    prepare_finalize = object.__new__(prepare_finalize_cls)
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=True,
    )

    with pytest.raises(
        NotImplementedError,
        match="does not support activation quantization in Prepare",
    ):
        prepare_finalize.prepare_async(
            a1=torch.zeros((1, 4), dtype=torch.bfloat16),
            topk_weights=torch.ones((1, 1), dtype=torch.float32),
            topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            num_experts=1,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
            defer_input_quant=False,
        )


def _moe(
    *,
    backend="mpi_alltoallv_v2",
    num_experts=4,
    num_local_experts=2,
    enable_eplb=False,
    is_sequence_parallel=False,
):
    return SimpleNamespace(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        max_num_tokens=16,
        tp_rank=0,
        tp_size=1,
        dp_rank=0,
        dp_size=2,
        moe_parallel_config=SimpleNamespace(
            all2all_backend=backend,
            use_all2all_kernels=True,
            enable_eplb=enable_eplb,
            ep_size=2,
            tp_rank=0,
            tp_size=1,
            sp_size=2 if is_sequence_parallel else 1,
            is_sequence_parallel=is_sequence_parallel,
        ),
    )


def test_torch_alltoall_prepare_keeps_router_weights_on_source_rank(monkeypatch):
    prepare_finalize = object.__new__(TorchAlltoallSinglePrepareAndFinalize)
    prepare_finalize.ep_group = object()
    prepare_finalize.ep_size = 1
    prepare_finalize.num_local_experts = 2
    prepare_finalize._send_split_sizes = None
    prepare_finalize._recv_split_sizes = None
    communicated_dtypes = []

    def all_to_all_single(output, input, **kwargs):
        communicated_dtypes.append(input.dtype)
        output.copy_(input)

    monkeypatch.setattr(torch_a2a.dist, "all_to_all_single", all_to_all_single)
    monkeypatch.setattr(
        torch_a2a,
        "count_expert_num_tokens",
        lambda topk_ids, num_local_experts, expert_map: torch.tensor(
            [1, 1], dtype=torch.int32
        ),
    )

    original_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    result = prepare_finalize.prepare(
        a1=torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        topk_weights=original_weights,
        topk_ids=torch.tensor([[0, 1]], dtype=torch.int64),
        num_experts=2,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=None,
    )
    _, _, _, dispatched_ids, dispatched_weights = result

    assert communicated_dtypes == [torch.int64, torch.bfloat16, torch.int64]
    assert prepare_finalize._topk_weights is original_weights
    assert dispatched_weights.shape == dispatched_ids.shape == (2, 1)
    assert dispatched_weights.dtype == original_weights.dtype


def test_torch_alltoall_uses_saved_weights_after_reverse_communication(monkeypatch):
    prepare_finalize = object.__new__(TorchAlltoallSinglePrepareAndFinalize)
    prepare_finalize.ep_group = object()
    prepare_finalize._recv_split_sizes = [2]
    prepare_finalize._send_split_sizes = [2]
    prepare_finalize._row_indices_restore = torch.tensor([0, 0])
    prepare_finalize._sort_indices = torch.tensor([1, 0])
    prepare_finalize._topk_weights = torch.tensor([[0.25, 0.75]])
    reverse_input = None

    def all_to_all_single(output, input, **kwargs):
        nonlocal reverse_input
        reverse_input = input.clone()
        output.copy_(input)

    monkeypatch.setattr(torch_a2a.dist, "all_to_all_single", all_to_all_single)

    output = torch.empty((1, 2), dtype=torch.float32)
    prepare_finalize.finalize(
        output=output,
        fused_expert_output=torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
        topk_weights=torch.full((2, 1), torch.nan),
        topk_ids=torch.tensor([[0], [1]]),
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=TopKWeightAndReduceDelegate(),
    )

    torch.testing.assert_close(reverse_input, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))
    torch.testing.assert_close(output, torch.tensor([[2.5, 3.5]]))
    assert prepare_finalize._row_indices_restore is None
    assert prepare_finalize._sort_indices is None
    assert prepare_finalize._topk_weights is None


def test_mpi_v1_fails_before_group_access(monkeypatch):
    monkeypatch.setattr(
        pf_factory,
        "get_ep_group",
        lambda: pytest.fail("deprecated v1 must fail before group access"),
    )

    with pytest.raises(ValueError, match=r"v1.*deprecated.*mpi_alltoallv_v2"):
        pf_factory.MpiAlltoallvV1PrepareFinalizeFactory.create(
            # Deprecation takes precedence over unrelated layout validation.
            moe=_moe(num_experts=5),
            quant_config=None,
            routing_tables=None,
            allow_new_interface=True,
            use_monolithic=False,
            eep_stage=False,
            all2all_manager=object(),
        )


def test_mpi_v2_rejects_sequence_parallel_before_group_access(monkeypatch):
    monkeypatch.setattr(
        pf_factory,
        "get_ep_group",
        lambda: pytest.fail("invalid v2 config must fail before group access"),
    )

    with pytest.raises(ValueError, match="v2 does not support sequence parallelism"):
        pf_factory.MpiAlltoallvV2PrepareFinalizeFactory.create(
            moe=_moe(backend="mpi_alltoallv_v2", is_sequence_parallel=True),
            quant_config=None,
            routing_tables=None,
            allow_new_interface=True,
            use_monolithic=False,
            eep_stage=False,
            all2all_manager=object(),
        )


@pytest.mark.parametrize(
    ("prepare_finalize_cls", "op_name"),
    [
        (MpiAlltoallvPrepareAndFinalizeV3, "moe_prepare_fused_v3"),
        (MpiAlltoallvPrepareAndFinalizeV4, "moe_prepare_fused_v4"),
    ],
)
def test_mpi_v3_v4_prepare_sanitizes_fp8_route_inputs(
    monkeypatch,
    prepare_finalize_cls,
    op_name,
):
    import torch_xcpu

    prepare_finalize = object.__new__(prepare_finalize_cls)
    prepare_finalize.max_num_tokens = 4
    prepare_finalize.max_moe_tokens_per_rank = 4
    prepare_finalize.num_local_experts = 2
    prepare_finalize.rank_expert_offset = 2
    prepare_finalize.ep_size = 2
    prepare_finalize.dp_size = 2
    prepare_finalize._comm_metadata = torch.zeros(6, dtype=torch.int64)
    prepare_finalize.comm_ptr_wrapper = object()

    if prepare_finalize_cls is MpiAlltoallvPrepareAndFinalizeV4:
        prepare_finalize._get_static_buffer_size = lambda topk, hidden_dim: 10
        prepare_finalize._check_single_sender_capacity = (
            lambda topk, static_buffer_size: None
        )
        prepare_finalize._check_recv_buffer_capacity = (
            lambda topk_ids, num_experts, static_buffer_size: None
        )

    def prepare_op(*args):
        recv_topk_ids = args[2]
        expert_num_tokens = args[3]
        num_input_rows_valid = args[4]
        recv_topk_ids[0] = 1
        expert_num_tokens.zero_()
        expert_num_tokens[2:] = torch.tensor([3, 5], dtype=torch.int32)
        num_input_rows_valid.fill_(8)

    monkeypatch.setattr(torch_xcpu.ops, op_name, prepare_op)
    prepare_result = prepare_finalize.prepare(
        a1=torch.zeros((1, 2), dtype=torch.bfloat16),
        topk_weights=torch.tensor([[0.25]], dtype=torch.float32),
        topk_ids=torch.tensor([[1]], dtype=torch.int64),
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        defer_input_quant=True,
    )

    _, _, expert_tokens_meta, prepared_ids, prepared_weights = prepare_result
    assert prepared_ids[0, 0].item() == 1
    assert torch.all(prepared_ids[1:] == -1)
    torch.testing.assert_close(
        expert_tokens_meta.expert_num_tokens,
        torch.tensor([3, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_tokens_meta.num_input_rows_valid,
        torch.tensor([8], dtype=torch.int32),
    )
    assert prepared_weights.shape == prepared_ids.shape
    assert prepared_weights.dtype == torch.float32


def test_mpi_v3_reserves_a_worst_case_segment_for_each_sender():
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV3)
    prepare_finalize.max_moe_tokens_per_rank = 32
    prepare_finalize.num_local_experts = 64
    prepare_finalize.ep_size = 4

    # Each of four senders may route all 32 * topk rows to this rank. The
    # extra row in every segment stores aligned per-expert metadata.
    assert prepare_finalize._get_static_buffer_size(topk=8, hidden_dim=2048) == (
        4 * (32 * 8 + 1)
    )


def test_mpi_v3_sequence_parallel_capacity_uses_local_token_bound():
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV3)
    prepare_finalize.max_moe_tokens_per_rank = 17
    prepare_finalize.num_local_experts = 64
    prepare_finalize.ep_size = 4

    assert prepare_finalize._get_static_buffer_size(topk=8, hidden_dim=2048) == (
        4 * (17 * 8 + 1)
    )


@pytest.mark.parametrize(
    ("moe", "routing_tables", "match"),
    [
        (_moe(num_experts=5), None, "uniform expert partition"),
        (_moe(enable_eplb=True), None, "linear placement without EPLB"),
        (_moe(), (object(), object(), object()), "linear placement without EPLB"),
    ],
)
def test_custom_backend_rejects_unsupported_layout_before_group_init(
    monkeypatch,
    moe,
    routing_tables,
    match,
):
    monkeypatch.setattr(
        pf_factory,
        "get_ep_group",
        lambda: pytest.fail("group initialization must not run"),
    )

    with pytest.raises(NotImplementedError, match=match):
        pf_factory.MpiAlltoallvV2PrepareFinalizeFactory.create(
            moe=moe,
            quant_config=None,
            routing_tables=routing_tables,
            allow_new_interface=True,
            use_monolithic=False,
            eep_stage=False,
            all2all_manager=object(),
        )


def test_custom_backend_rejects_dbo_before_group_init(monkeypatch):
    monkeypatch.setattr(
        pf_factory,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(enable_dbo=True),
        ),
    )
    monkeypatch.setattr(
        pf_factory,
        "get_ep_group",
        lambda: pytest.fail("group initialization must not run"),
    )

    with pytest.raises(NotImplementedError, match="does not support DBO"):
        pf_factory.MpiAlltoallvV2PrepareFinalizeFactory.create(
            moe=_moe(),
            quant_config=None,
            routing_tables=None,
            allow_new_interface=True,
            use_monolithic=False,
            eep_stage=False,
            all2all_manager=object(),
        )

