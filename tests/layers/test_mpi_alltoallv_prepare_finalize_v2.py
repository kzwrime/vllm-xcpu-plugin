from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)

from vllm_xcpu_plugin.layers.fused_moe import (
    mpi_alltoallv_prepare_finalize_v2 as mpi_v2,
)
from vllm_xcpu_plugin.layers.fused_moe.cpu_groupgemm_moe_v2 import (
    CPUGroupGemmExperts,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v2 import (
    MpiAlltoallvPrepareAndFinalizeV2,
)


def _prepare_finalize():
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV2)
    prepare_finalize.max_num_tokens = 4
    prepare_finalize.num_local_experts = 2
    prepare_finalize.rank_expert_offset = 2
    prepare_finalize.dp_rank = 0
    prepare_finalize.dp_size = 1
    prepare_finalize.tp_rank = 0
    prepare_finalize.tp_size = 1
    prepare_finalize.ep_rank = 1
    prepare_finalize.ep_size = 2
    prepare_finalize.num_dispatchers_ = 2
    prepare_finalize._comm_metadata = torch.tensor(
        [2, 1, 0, 1, 0, 1], dtype=torch.int64
    )
    prepare_finalize.comm_ptr_wrapper = torch.tensor([0], dtype=torch.int64)
    prepare_finalize._sort_indices_back = None
    prepare_finalize._full_send_split_sizes = None
    prepare_finalize._recv_split_sizes = None
    prepare_finalize._topk_weights = None
    prepare_finalize.topk = -1
    return prepare_finalize


def test_v2_reads_model_tp_coordinates_internally(monkeypatch):
    class Communicator:
        comm_ptr_wrapper = torch.tensor([0], dtype=torch.int64)

    monkeypatch.setattr(mpi_v2, "CpuMPICommunicator", Communicator)
    monkeypatch.setattr(
        mpi_v2,
        "get_ep_group",
        lambda: SimpleNamespace(device_communicator=Communicator()),
    )
    monkeypatch.setattr(mpi_v2.dist, "get_rank", lambda group: 3)
    monkeypatch.setattr(mpi_v2.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(mpi_v2, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        mpi_v2, "get_tensor_model_parallel_world_size", lambda: 2
    )

    prepare_finalize = MpiAlltoallvPrepareAndFinalizeV2(
        max_num_tokens=16,
        ep_group=object(),
        num_experts=8,
        num_local_experts=2,
        num_dispatchers=4,
        rank_expert_offset=6,
        dp_rank=1,
        dp_size=2,
    )

    assert prepare_finalize.tp_rank == 1
    assert prepare_finalize.tp_size == 2
    torch.testing.assert_close(
        prepare_finalize._comm_metadata,
        torch.tensor([4, 3, 1, 2, 1, 2], dtype=torch.int64),
    )


def test_unquantized_route_output_delegates_weight_and_reduce():
    experts = CPUGroupGemmExperts(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=True),
        ),
        quant_config=object(),
    )
    assert isinstance(
        experts.finalize_weight_and_reduce_impl(),
        TopKWeightAndReduceDelegate,
    )
    assert experts.workspace_shapes(
        M=3,
        N=8,
        K=4,
        topk=2,
        global_num_experts=4,
        local_num_experts=2,
        expert_tokens_meta=None,
        activation=object(),
    )[-1] == (6, 4)


def test_unquantized_no_ep_reduces_topk_inside_experts():
    experts = CPUGroupGemmExperts(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=False),
        ),
        quant_config=object(),
    )
    assert isinstance(
        experts.finalize_weight_and_reduce_impl(),
        TopKWeightAndReduceNoOP,
    )
    assert experts.workspace_shapes(
        M=3,
        N=8,
        K=4,
        topk=2,
        global_num_experts=4,
        local_num_experts=4,
        expert_tokens_meta=None,
        activation=object(),
    )[-1] == (3, 4)


def test_unquantized_mpi_v5_reduces_destination_local_routes():
    experts = CPUGroupGemmExperts(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_ep=True,
                all2all_backend="mpi_alltoallv_v5",
            ),
        ),
        quant_config=object(),
    )
    assert isinstance(
        experts.finalize_weight_and_reduce_impl(),
        TopKWeightAndReduceNoOP,
    )
    assert experts.workspace_shapes(
        M=3,
        N=8,
        K=4,
        topk=2,
        global_num_experts=4,
        local_num_experts=2,
        expert_tokens_meta=None,
        activation=object(),
    )[-1] == (3, 4)


@pytest.mark.parametrize(
    ("use_ep", "expected_topk_reduce", "expected_output_rows"),
    [(False, True, 2), (True, False, 4)],
)
def test_unquantized_experts_select_kernel_reduce_from_ep_mode(
    monkeypatch,
    use_ep,
    expected_topk_reduce,
    expected_output_rows,
):
    from torch_xcpu import ops as xcpu_ops

    captured = {}

    def fused_moe_compute(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(xcpu_ops, "fused_moe_compute", fused_moe_compute)
    experts = CPUGroupGemmExperts(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    output = torch.empty((expected_output_rows, 4), dtype=torch.bfloat16)
    experts.apply(
        output=output,
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=torch.ones((1, 8, 4), dtype=torch.bfloat16),
        w2=torch.ones((1, 4, 4), dtype=torch.bfloat16),
        topk_weights=torch.full((2, 2), 0.5, dtype=torch.float32),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=1,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured["topk_reduce"] is expected_topk_reduce
    expected_workspace_shape = (2, 4) if expected_topk_reduce else (0,)
    assert captured["workspace_unpermute_and_reduce"].shape == (
        expected_workspace_shape
    )


def test_finalize_requires_delegate_and_uses_original_router_weights(
    monkeypatch,
):
    from torch_xcpu import ops as xcpu_ops

    prepare_finalize = _prepare_finalize()
    prepare_finalize.topk = 1
    prepare_finalize._sort_indices_back = torch.zeros(1, dtype=torch.int32)
    prepare_finalize._recv_split_sizes = torch.ones(2, dtype=torch.int32)
    prepare_finalize._full_send_split_sizes = torch.ones(2, dtype=torch.int32)
    original_weights = torch.tensor([[0.25]], dtype=torch.float32)
    prepare_finalize._topk_weights = original_weights
    captured = {}

    def finalize_op(*args):
        captured["router_weights"] = args[-1]

    monkeypatch.setattr(xcpu_ops, "moe_finalize", finalize_op)
    prepare_finalize.finalize(
        output=torch.empty((1, 4), dtype=torch.bfloat16),
        fused_expert_output=torch.empty((2, 4), dtype=torch.bfloat16),
        topk_weights=torch.ones((2, 1), dtype=torch.float32),
        topk_ids=torch.tensor([[2], [-1]], dtype=torch.int32),
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=TopKWeightAndReduceDelegate(),
    )

    assert captured["router_weights"] is original_weights
