"""Command-line entry point."""

import argparse
import os


def main():
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(
        description="LOGOS synthetic finite-course research workflow"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate wholly synthetic observations")
    gen.add_argument("--output", required=True)
    gen.add_argument("--profiles", type=int, default=4)
    gen.add_argument("--seed", type=int, default=20261012)
    run = sub.add_parser("run", help="Fit, optimize, verify and plot synthetic data")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        from .synthetic import write_dataset

        write_dataset(args.output, args.profiles, args.seed)
    else:
        from .pipeline import run as execute

        execute(args.data, args.output)


if __name__ == "__main__":
    main()
