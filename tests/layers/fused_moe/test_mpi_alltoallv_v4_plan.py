# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_prepare_finalize_v4 import (
    MpiAlltoallvPrepareAndFinalizeV4,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_v4_plan import (
    compute_send_rounds,
    validate_single_sender_capacity,
)


def test_compute_send_rounds_all_zero():
    counts = torch.zeros((4, 4), dtype=torch.int64)

    assert compute_send_rounds(counts, max_recv_tokens=1) == [(0, 4)]


def test_compute_send_rounds_exact_limit():
    counts = torch.tensor(
        [
            [2, 1, 0],
            [1, 2, 3],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )

    assert compute_send_rounds(counts, max_recv_tokens=4) == [(0, 3)]


def test_compute_send_rounds_single_hot_receiver():
    counts = torch.tensor(
        [
            [0, 3, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.int64,
    )

    assert compute_send_rounds(counts, max_recv_tokens=5) == [(0, 2), (2, 4)]


def test_compute_send_rounds_single_hot_sender():
    counts = torch.tensor(
        [
            [0, 0, 0],
            [2, 3, 1],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )

    assert compute_send_rounds(counts, max_recv_tokens=3) == [(0, 3)]


def test_compute_send_rounds_uniform():
    counts = torch.ones((4, 4), dtype=torch.int64)

    assert compute_send_rounds(counts, max_recv_tokens=2) == [(0, 2), (2, 4)]


def test_compute_send_rounds_single_sender_exceeds_limit():
    counts = torch.tensor(
        [
            [0, 7],
            [0, 0],
        ],
        dtype=torch.int64,
    )

    with pytest.raises(RuntimeError, match="single sender exceeds"):
        compute_send_rounds(counts, max_recv_tokens=6)


def test_validate_single_sender_capacity_passes_at_exact_limit():
    validate_single_sender_capacity(
        topk=8,
        max_num_local_experts=4,
        max_tokens=4096,
        max_recv_tokens=16384,
    )


def test_validate_single_sender_capacity_fails_when_worst_sender_exceeds():
    with pytest.raises(RuntimeError, match="single sender rank"):
        validate_single_sender_capacity(
            topk=8,
            max_num_local_experts=32,
            max_tokens=4096,
            max_recv_tokens=8192,
        )


def test_default_v4_buffer_covers_all_ep_senders():
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV4)
    prepare_finalize.max_recv_tokens = 0
    prepare_finalize.ep_size = 4
    prepare_finalize.max_moe_tokens_per_rank = 16
    prepare_finalize.num_local_experts = 64

    assert prepare_finalize._get_static_buffer_size(topk=8, hidden_dim=2048) == 512


def test_v4_reports_sequence_parallel_per_rank_token_capacity():
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV4)
    prepare_finalize.max_num_tokens = 32
    prepare_finalize.max_moe_tokens_per_rank = 16

    assert prepare_finalize.max_num_tokens_per_rank() == 16


def test_collect_send_counts_ignores_padding_expert_ids():
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV4)
    prepare_finalize.ep_size = 2
    prepare_finalize.num_local_experts_ranks = torch.tensor([2, 2])
    prepare_finalize.ep_group_coordinator = type(
        "Coordinator",
        (),
        {"all_gather": lambda self, value, dim: torch.cat((value, value), dim=dim)},
    )()

    actual = prepare_finalize._collect_send_count_overall(
        torch.tensor([[0, -1], [3, -1]]),
        num_experts=4,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[1, 1], [1, 1]], dtype=torch.int64),
    )


@pytest.mark.parametrize("invalid_id", [-2, 4])
def test_collect_send_counts_rejects_invalid_expert_ids(invalid_id):
    prepare_finalize = object.__new__(MpiAlltoallvPrepareAndFinalizeV4)
    prepare_finalize.ep_size = 2
    prepare_finalize.num_local_experts_ranks = torch.tensor([2, 2])

    with pytest.raises(ValueError, match=f"Invalid expert ID {invalid_id}"):
        prepare_finalize._collect_send_count_overall(
            torch.tensor([[invalid_id]]),
            num_experts=4,
        )
