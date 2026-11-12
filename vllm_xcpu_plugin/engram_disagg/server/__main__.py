"""Command-line entry point for the Engram-rank MPMD program."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XCPU Engram disaggregation server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--load-format",
        default="auto",
        choices=("auto", "safetensors", "dummy"),
        help="dummy fills the tables with the constants vLLM's dummy loader uses",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .runner import EngramServerOptions, run_engram_server

    run_engram_server(EngramServerOptions(**vars(args)))


if __name__ == "__main__":
    main()
