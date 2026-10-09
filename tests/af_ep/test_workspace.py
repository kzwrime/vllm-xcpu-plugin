import pytest
import torch


@pytest.mark.parametrize("version", [7, 8])
@pytest.mark.parametrize("sp,expected_matrix", [(1, 336), (2, 168), (4, 84)])
def test_af_workspace_compact_capacity(version, sp, expected_matrix):
    import torch_xcpu  # noqa: F401

    query = getattr(
        torch.ops.torch_xcpu,
        f"fused_af_f_moe_v{version}_workspace_size_AccFp8MoeGroupedGemm",
    )
    rows = 1024 // sp
    n = query(4, rows, 8, 6144, 2048, 4, 2, 128, 128, False, False)
    metadata = 4 * rows * 8 * 16 + 4 * 64
    assert n == expected_matrix * 1024**2 + metadata
    with pytest.raises(RuntimeError, match="workspace dimensions"):
        query(4, 0, 8, 6144, 2048, 4, 2, 128, 128, False, False)


@pytest.mark.parametrize("version", [7, 8])
def test_f_workspace_rejected_before_communication(version):
    import torch_mcpu  # noqa: F401
    import torch_xcpu  # noqa: F401

    op = getattr(
        torch.ops.torch_xcpu, f"fused_af_f_moe_v{version}_PortableBf16MoeGroupedGemm"
    )
    w1 = torch.zeros(4, 32, 16, dtype=torch.bfloat16, device="mcpu")
    w2 = torch.zeros(4, 16, 16, dtype=torch.bfloat16, device="mcpu")
    expert_map = torch.tensor(
        [0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32, device="mcpu"
    )
    # Intentionally invalid communicator: rejection must happen before an MPI call.
    metadata = torch.tensor(
        [1 if version == 7 else 2, 4, 2, 2, 2, 2, 1, 0], dtype=torch.int64, device="cpu"
    )
    comm = torch.tensor([-1], dtype=torch.int64, device="cpu")
    prefix = (
        w1,
        w2,
        None,
        None,
        None,
        None,
        0,
        0,
        8,
        expert_map,
        8,
        8,
        2,
        1,
        metadata,
        comm,
    )
    with pytest.raises(RuntimeError, match="workspace too small"):
        op(*prefix, torch.empty(64, dtype=torch.uint8, device="mcpu"))
    backing = torch.empty(1025, dtype=torch.uint8, device="mcpu")
    with pytest.raises(RuntimeError, match="aligned"):
        op(*prefix, backing[1:])


@pytest.mark.parametrize("version", [7, 8])
def test_attention_workspace_rejected_before_communication(version):
    import torch_mcpu  # noqa: F401
    import torch_xcpu  # noqa: F401

    op = getattr(torch.ops.torch_xcpu, f"fused_af_a_dispatch_combine_v{version}_bf16")
    x = torch.zeros(2, 16, dtype=torch.bfloat16, device="mcpu")
    ids = torch.zeros(2, 8, dtype=torch.int32, device="mcpu")
    weights = torch.ones(2, 8, device="mcpu")
    metadata = torch.tensor(
        [1 if version == 7 else 2, 4, 0, 2, 2, 2, 0, 0], dtype=torch.int64, device="cpu"
    )
    comm = torch.tensor([-1], dtype=torch.int64, device="cpu")
    workspace = torch.empty(64, dtype=torch.uint8, device="mcpu")
    with pytest.raises(RuntimeError, match="workspace too small"):
        op(torch.empty_like(x), x, ids, weights, 8, 8, 1, metadata, comm, workspace)
    with pytest.raises(RuntimeError, match="alias"):
        op(x, x, ids, weights, 8, 8, 1, metadata, comm, workspace)
