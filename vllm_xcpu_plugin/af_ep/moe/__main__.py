"""Command-line entry point for the AF-EP F-rank MPMD program."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XCPU AF-EP expert service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument(
        "--max-model-passes",
        type=int,
        default=0,
        help="0 serves until the MPMD launcher terminates the process",
    )
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction,
                        default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .runner import ExpertServiceOptions, run_expert_service

    run_expert_service(ExpertServiceOptions(**vars(args)))


if __name__ == "__main__":
    main()
