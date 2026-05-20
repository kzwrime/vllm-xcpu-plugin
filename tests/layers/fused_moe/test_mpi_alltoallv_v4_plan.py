# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
