import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from vllm.config import parallel as parallel_config_module
from vllm.model_executor.layers.fused_moe import all2all_utils
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)

import vllm_xcpu_plugin.layers.fused_moe.prepare_finalize_factory as pf_factory
from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import (
    MPI_ALLTOALLV_BACKENDS,
)
from vllm_xcpu_plugin.layers.fused_moe import (
    torch_all_to_all_single_prepare_finalize as torch_a2a,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v1 import (
    MpiAlltoallvPrepareAndFinalizeV1,
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


def test_xcpu_registers_versioned_vllm_prepare_finalize_backends():
    assert {
        backend: all2all_utils._PREPARE_FINALIZE_FACTORIES[backend]
        for backend in (
            "torch_all_to_all_single",
            "mpi_alltoallv_v1",
            "mpi_alltoallv_v2",
            "mpi_alltoallv_v3",
            "mpi_alltoallv_v4",
            "mpi_alltoallv_v5",
            "mpi_alltoallv",
        )
    } == {
        "torch_all_to_all_single": (
            pf_factory.TorchAllToAllSinglePrepareFinalizeFactory
        ),
        "mpi_alltoallv_v1": pf_factory.MpiAlltoallvV1PrepareFinalizeFactory,
        "mpi_alltoallv_v2": pf_factory.MpiAlltoallvV2PrepareFinalizeFactory,
        "mpi_alltoallv_v3": pf_factory.MpiAlltoallvV3PrepareFinalizeFactory,
        "mpi_alltoallv_v4": pf_factory.MpiAlltoallvV4PrepareFinalizeFactory,
        "mpi_alltoallv_v5": pf_factory.MpiAlltoallvV5PrepareFinalizeFactory,
        "mpi_alltoallv": pf_factory.MpiAlltoallvLegacyPrepareFinalizeFactory,
    }
    assert {
        "torch_all_to_all_single",
        "mpi_alltoallv_v3",
        "mpi_alltoallv_v4",
        "mpi_alltoallv_v5",
    } <= parallel_config_module.SEQUENCE_PARALLEL_MOE_BACKENDS
    assert {
        "mpi_alltoallv_v1",
        "mpi_alltoallv_v2",
        "mpi_alltoallv",
    }.isdisjoint(parallel_config_module.SEQUENCE_PARALLEL_MOE_BACKENDS)
    assert "mpi_alltoallv_v5" in MPI_ALLTOALLV_BACKENDS


def test_worker_plugin_flow_registers_moe_runtime_extensions():
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "xcpu_platform_plugin,xcpu_custom_ops"
    code = """
import os
from types import SimpleNamespace

from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins

assert current_platform.device_name == "mcpu"
load_general_plugins()

from vllm.config import ParallelConfig
from vllm.model_executor.layers.fused_moe import all2all_utils

assert set(all2all_utils._PREPARE_FINALIZE_FACTORIES) == {
    "mpi_alltoallv",
    "mpi_alltoallv_v1",
    "mpi_alltoallv_v2",
    "mpi_alltoallv_v3",
    "mpi_alltoallv_v4",
    "mpi_alltoallv_v5",
    "torch_all_to_all_single",
}
config = ParallelConfig(
    all2all_backend="mpi_alltoallv_v2",
    enable_expert_parallel=True,
    tensor_parallel_size=2,
    data_parallel_size=2,
)
assert not config.use_sequence_parallel_moe

v3_sp_config = ParallelConfig(
    all2all_backend="mpi_alltoallv_v3",
    enable_expert_parallel=True,
    tensor_parallel_size=2,
    data_parallel_size=2,
)
assert v3_sp_config.use_sequence_parallel_moe

moe = SimpleNamespace(
    moe_parallel_config=SimpleNamespace(
        all2all_backend="mpi_alltoallv_v1",
        use_all2all_kernels=True,
    ),
)
all2all_utils.get_ep_all2all_manager = lambda eep_stage: SimpleNamespace()
try:
    all2all_utils.maybe_make_prepare_finalize(
        moe=moe,
        quant_config=None,
        allow_new_interface=True,
    )
except ValueError as exc:
    assert "v1 is deprecated and no longer supported" in str(exc)
    assert "mpi_alltoallv_v2" in str(exc)
else:
    raise AssertionError("MPI alltoallv v1 must fail through the registered factory")

sp_config = ParallelConfig(
    all2all_backend="mpi_alltoallv_v4",
    enable_expert_parallel=True,
    tensor_parallel_size=2,
    data_parallel_size=2,
)
assert sp_config.use_sequence_parallel_moe

v5_sp_config = ParallelConfig(
    all2all_backend="mpi_alltoallv_v5",
    enable_expert_parallel=True,
    tensor_parallel_size=2,
    data_parallel_size=2,
)
assert v5_sp_config.use_sequence_parallel_moe
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "prepare_finalize_cls",
    [
        TorchAlltoallSinglePrepareAndFinalize,
        MpiAlltoallvPrepareAndFinalizeV2,
        MpiAlltoallvPrepareAndFinalizeV3,
        MpiAlltoallvPrepareAndFinalizeV4,
        MpiAlltoallvPrepareAndFinalizeV5,
    ],
)
def test_prepare_finalize_does_not_publish_xcpu_expert_capability_flags(
    prepare_finalize_cls,
):
    assert "xcpu_experts_reduce_topk" not in prepare_finalize_cls.__dict__
    assert "xcpu_delegate_weight_and_reduce" not in prepare_finalize_cls.__dict__


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


def test_mpi_v1_cannot_be_instantiated():
    with pytest.raises(
        RuntimeError,
        match=r"deprecated.*no longer supported.*v2, v3, v4, or v5",
    ):
        MpiAlltoallvPrepareAndFinalizeV1(
            max_num_tokens=1,
            ep_group=None,
            num_experts=1,
            num_local_experts=1,
            num_dispatchers=1,
            rank_expert_offset=0,
            dp_rank=0,
            dp_size=1,
        )


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
    ("factory", "is_sequence_parallel"),
    [
        (pf_factory.MpiAlltoallvV2PrepareFinalizeFactory, False),
        (pf_factory.MpiAlltoallvV3PrepareFinalizeFactory, False),
        (pf_factory.MpiAlltoallvV3PrepareFinalizeFactory, True),
        (pf_factory.MpiAlltoallvV4PrepareFinalizeFactory, False),
        (pf_factory.MpiAlltoallvV4PrepareFinalizeFactory, True),
        (pf_factory.MpiAlltoallvV5PrepareFinalizeFactory, False),
        (pf_factory.MpiAlltoallvV5PrepareFinalizeFactory, True),
    ],
)
def test_mpi_v2_v3_v4_v5_build_from_registered_factory(
    monkeypatch,
    factory,
    is_sequence_parallel,
):
    ep_group = object()
    all2all_manager = SimpleNamespace(world_size=4, rank=2, cpu_group=object())
    captured = {}

    class PrepareFinalize:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(factory, "implementation", PrepareFinalize)
    monkeypatch.setattr(
        pf_factory,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=ep_group),
    )

    actual = factory.create(
        moe=_moe(
            backend=factory.backend_name,
            is_sequence_parallel=is_sequence_parallel,
        ),
        quant_config=None,
        routing_tables=None,
        allow_new_interface=True,
        use_monolithic=False,
        eep_stage=False,
        all2all_manager=all2all_manager,
    )

    assert isinstance(actual, PrepareFinalize)
    assert captured["ep_group"] is ep_group
    assert captured["rank_expert_offset"] == 4
    assert "tp_rank" not in captured
    assert "tp_size" not in captured
    assert captured.get("is_sequence_parallel", False) is is_sequence_parallel
    assert captured.get("sp_size", 1) == (2 if is_sequence_parallel else 1)


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


def test_torch_backend_builds_from_vllm_registered_factory(monkeypatch):
    ep_group = object()
    all2all_manager = SimpleNamespace(world_size=4, rank=2, cpu_group=object())
    captured = {}

    class PrepareFinalize:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        pf_factory,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=ep_group),
    )
    monkeypatch.setattr(
        pf_factory,
        "TorchAlltoallSinglePrepareAndFinalize",
        PrepareFinalize,
    )

    actual = pf_factory.TorchAllToAllSinglePrepareFinalizeFactory.create(
        moe=_moe(
            backend="torch_all_to_all_single",
            num_experts=12,
            num_local_experts=6,
        ),
        quant_config=None,
        routing_tables=None,
        allow_new_interface=True,
        use_monolithic=False,
        eep_stage=False,
        all2all_manager=all2all_manager,
    )

    assert isinstance(actual, PrepareFinalize)
    assert captured["ep_group"] is ep_group
    assert captured == {
        "ep_group": ep_group,
        "num_local_experts": 6,
        "num_dispatchers": 4,
    }
