import pytest

from vllm_xcpu_plugin.distributed.mpi_world import (
    ClusterType,
    _build_cluster_directory,
)


def test_cluster_directory_preserves_local_rank_order() -> None:
    clusters = _build_cluster_directory([
        (0, 0, 0, 2),
        (0, 0, 1, 2),
        (1, 0, 1, 2),
        (1, 0, 0, 2),
        (2, 1, 0, 1),
    ])

    assert clusters[0].cluster_type == ClusterType.ATTN
    assert clusters[0].global_ranks == (0, 1)
    assert clusters[1].cluster_type == ClusterType.ATTN
    assert clusters[1].global_ranks == (3, 2)
    assert clusters[2].cluster_type == ClusterType.MOE
    assert clusters[2].global_ranks == (4,)


@pytest.mark.parametrize(
    "descriptors",
    [
        [(0, 0, 0, 2)],
        [(0, 0, 0, 2), (0, 0, 0, 2)],
        [(0, 0, 0, 2), (0, 0, 1, 3)],
        [(0, 0, 0, 2), (0, 1, 1, 2)],
        [(0, 2, 0, 1)],
    ],
)
def test_cluster_directory_rejects_inconsistent_membership(descriptors) -> None:
    with pytest.raises((AssertionError, ValueError)):
        _build_cluster_directory(descriptors)

