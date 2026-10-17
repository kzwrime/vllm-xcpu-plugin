"""Command-line entry point for the AF-EP F-rank MPMD program."""

from __future__ import annotations

import argparse

from ..common import AF_EP_BACKENDS, DEFAULT_AF_EP_BACKEND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XCPU AF-EP expert service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--all2all-backend",
        choices=AF_EP_BACKENDS,
        default=DEFAULT_AF_EP_BACKEND,
    )
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument(
        "--max-model-passes",
        type=int,
        default=0,
        help="0 serves until the MPMD launcher terminates the process",
    )
    parser.add_argument("--load-format", default="auto")
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="compile each receive/experts/send transaction with Inductor",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .runner import ExpertServiceOptions, run_expert_service

    run_expert_service(ExpertServiceOptions(**vars(args)))


if __name__ == "__main__":
    main()
