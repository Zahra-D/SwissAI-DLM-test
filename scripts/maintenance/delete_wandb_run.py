#!/usr/bin/env python3
"""Delete one explicitly named W&B run after checking its expected metadata."""

import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()
    run = wandb.Api().run(args.run)
    if run.summary.get("baseline/source_run") != args.expected_source:
        raise RuntimeError("refusing to delete: unexpected source run")
    if run.summary.get("baseline/history_rows_copied") != args.expected_rows:
        raise RuntimeError("refusing to delete: unexpected history-row count")
    print(f"deleting={run.path}")
    run.delete()


if __name__ == "__main__":
    main()
