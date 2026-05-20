# SPDX-License-Identifier: Apache-2.0

import torch


def validate_single_sender_capacity(
    topk: int,
    max_num_local_experts: int,
    max_tokens: int,
    max_recv_tokens: int,
) -> None:
    max_sender_tokens = min(topk, max_num_local_experts) * max_tokens
    if max_sender_tokens > max_recv_tokens:
        raise RuntimeError(
            "moe_prepare_fused_v4 receive token buffer is too small for a "
            "single sender rank: "
            f"min(topk={topk}, max_num_local_experts={max_num_local_experts}) "
            f"* max_tokens={max_tokens} gives max_sender_tokens="
            f"{max_sender_tokens}, limit_tokens={max_recv_tokens}. "
            "Increase VLLM_XCPU_MOE_MAX_RECV_TOKENS or reduce "
            "--max-num-batched-tokens; sender-rank batching cannot make "
            "progress when one sender alone can exceed the buffer."
        )


def compute_send_rounds(
    send_count_overall: torch.Tensor,
    max_recv_tokens: int,
) -> list[tuple[int, int]]:
    """Partition sender ranks into deterministic contiguous alltoallv rounds.

    `send_count_overall[src, dst]` is the number of tokens sender rank `src`
    sends to receiver rank `dst`. Each returned tuple is a half-open sender
    range `[start_rank, end_rank)`. For every round and every receiver column,
    the token sum is guaranteed to be <= `max_recv_tokens`.
    """
    if max_recv_tokens <= 0:
        raise ValueError(
            "max_recv_tokens must be positive when computing MoE send rounds"
        )
    if send_count_overall.dim() != 2:
        raise ValueError("send_count_overall must be a 2-D tensor")

    ep_size, dst_size = send_count_overall.shape
    if ep_size == 0 or dst_size == 0:
        raise ValueError("send_count_overall must be non-empty")
    if ep_size != dst_size:
        raise ValueError(
            "send_count_overall must be square: "
            f"got shape={tuple(send_count_overall.shape)}"
        )

    counts = send_count_overall.to(device="cpu", dtype=torch.int64)
    if bool((counts < 0).any().item()):
        raise ValueError("send_count_overall must not contain negative counts")

    prefix = torch.empty((ep_size + 1, ep_size), dtype=torch.int64)
    prefix[0].zero_()
    prefix[1:] = torch.cumsum(counts, dim=0)

    rounds: list[tuple[int, int]] = []
    start = 0
    while start < ep_size:
        first_round_counts = prefix[start + 1] - prefix[start]
        if bool((first_round_counts > max_recv_tokens).any().item()):
            max_dst = int(torch.argmax(first_round_counts).item())
            max_tokens = int(first_round_counts[max_dst].item())
            raise RuntimeError(
                "moe_prepare_fused_v4 single sender exceeds receive token "
                "limit; sender-rank batching cannot make progress without "
                "splitting tokens inside the sender: "
                f"src_rank={start}, dst_rank={max_dst}, "
                f"send_tokens={max_tokens}, limit_tokens={max_recv_tokens}"
            )

        end = start + 1
        while end < ep_size:
            round_counts = prefix[end + 1] - prefix[start]
            if bool((round_counts > max_recv_tokens).any().item()):
                break
            end += 1

        rounds.append((start, end))
        start = end

    return rounds
