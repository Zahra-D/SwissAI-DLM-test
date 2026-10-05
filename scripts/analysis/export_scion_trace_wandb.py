#!/usr/bin/env python3
"""Export SCION trace metrics from W&B runs to a single step-level CSV."""

import argparse
from pathlib import Path

import pandas as pd
import wandb


DEFAULT_KEYS = [
    "_step",
    "trainer/loss",
    "run/train_loss",
    "stats/grad_norm_nuc_power_1",
    "stats/grad_norm_fro_power_1",
    "stats/local_smooth_spec",
    "stats/local_smooth_fro",
    "stats/num_nuc",
    "stats/den_spec",
    "rho/total",
    "rho/total_nuc",
    "rho/total_fro",
    "rho/noise_rho_sample_0",
    "rho/noise_rho_sample_1",
    "rho/noise_rho_sample_2",
    "rho/averaged_rho_over_samples",
    "rho/rho_over_averaged_norms",
    "rho/reference_samples",
    "rho/noise_step",
    "grad/noise_samples",
    "grad/noise_E_grad_norm2",
    "grad/noise_mean_grad_norm2",
    "grad/noise_sigma2",
    "grad/noise_sigma",
    "grad/noise_snr",
    "grad/noise_loss_mean",
    "grad/noise_loss_var",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", default=None)
    p.add_argument("--name_contains", default=None)
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--include_keys", nargs="*", default=DEFAULT_KEYS)
    p.add_argument(
        "--history_samples",
        type=int,
        default=100_000,
        help=(
            "Maximum history samples requested per run. Keep this above the "
            "largest max_steps so sparse rho/sigma events are not sampled away."
        ),
    )
    return p.parse_args()


def _nested(config, *path):
    value = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def main():
    args = parse_args()
    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    runs = api.runs(path)

    rows = []
    n = 0
    for run in runs:
        if args.group is not None and run.group != args.group:
            continue
        if args.name_contains is not None and args.name_contains not in (run.name or ""):
            continue

        cfg = run.config or {}
        hist = run.history(
            keys=args.include_keys,
            samples=args.history_samples,
            pandas=True,
        )
        if hist is None or hist.empty:
            continue

        if "_step" in hist.columns and "step" not in hist.columns:
            hist = hist.rename(columns={"_step": "step"})

        hist["run_id"] = run.id
        hist["run_name"] = run.name
        hist["run_group"] = run.group

        # GIDD names these n_blocks/hidden_size; older experiments used
        # n_layer/n_embd. Export one stable schema for both.
        hist["n_layer"] = (
            cfg.get("n_layer")
            or _nested(cfg, "model", "n_layer")
            or _nested(cfg, "model", "n_blocks")
        )
        hist["n_embd"] = (
            cfg.get("n_embd")
            or _nested(cfg, "model", "n_embd")
            or _nested(cfg, "model", "hidden_size")
        )
        hist["batch_size"] = (
            cfg.get("batch_size")
            or _nested(cfg, "loader", "global_batch_size")
        )
        hist["sequence_length"] = (
            cfg.get("sequence_length")
            or _nested(cfg, "model", "length")
        )
        hist["seed"] = cfg.get("seed")
        hist["max_steps"] = _nested(cfg, "trainer", "max_steps")

        rows.append(hist)
        n += 1
        if args.max_runs > 0 and n >= args.max_runs:
            break

    if not rows:
        raise RuntimeError("No runs matched filters; nothing to export.")

    out = pd.concat(rows, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out)} rows from {n} runs to {args.output_csv}")


if __name__ == "__main__":
    main()
