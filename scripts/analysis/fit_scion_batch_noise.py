#!/usr/bin/env python3
"""Fit fixed-model SCION rho(B) and gradient-noise sigma^2(B).

The W&B logger can repeat the last sparse noise value on ordinary training
steps.  This script first deduplicates rows by ``rho/noise_step`` so each
trace collection contributes exactly one observation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fit_scion_constants import _equation_string, _fit_shifted_power_law


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tail_events", type=int, default=2)
    parser.add_argument("--min_events", type=int, default=2)
    parser.add_argument("--robust_delta", type=float, default=0.15)
    parser.add_argument("--run_col", default="run_id")
    parser.add_argument("--batch_col", default="batch_size")
    parser.add_argument("--length_col", default="sequence_length")
    parser.add_argument("--event_col", default="rho/noise_step")
    parser.add_argument("--rho_col", default="rho/rho_over_averaged_norms")
    parser.add_argument("--sigma2_col", default="grad/noise_sigma2")
    return parser.parse_args()


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _relative_last_change(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(abs(values[-1] - values[-2]) / max(abs(values[-2]), 1e-12))


def aggregate_runs(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    event_rows = []

    for run_id, run in df.groupby(args.run_col, sort=False):
        required = [args.batch_col, args.event_col, args.rho_col, args.sigma2_col]
        if any(column not in run.columns for column in required):
            continue

        events = run.copy()
        for column in required[1:]:
            events[column] = pd.to_numeric(events[column], errors="coerce")
        events = events.dropna(subset=required[1:])
        events = events.sort_values(args.event_col).drop_duplicates(
            subset=[args.event_col], keep="last"
        )
        if len(events) < args.min_events:
            continue

        batch_values = _finite(run[args.batch_col])
        if batch_values.size == 0:
            continue
        batch_size = float(batch_values[-1])

        if args.length_col in run.columns:
            length_values = _finite(run[args.length_col])
        else:
            length_values = np.array([], dtype=float)
        sequence_length = float(length_values[-1]) if length_values.size else 1.0

        tail = events.tail(args.tail_events) if args.tail_events > 0 else events
        rho_values = _finite(tail[args.rho_col])
        sigma2_values = _finite(tail[args.sigma2_col])
        all_rho = _finite(events[args.rho_col])
        all_sigma2 = _finite(events[args.sigma2_col])
        if rho_values.size == 0 or sigma2_values.size == 0:
            continue

        summary_rows.append(
            {
                "run_id": str(run_id),
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "num_noise_events": int(len(events)),
                "first_noise_step": float(events[args.event_col].iloc[0]),
                "last_noise_step": float(events[args.event_col].iloc[-1]),
                "rho_hat": float(np.mean(rho_values)),
                "sigma2_hat": float(np.mean(sigma2_values)),
                "sigma_hat": float(math.sqrt(max(np.mean(sigma2_values), 0.0))),
                "sigma_star2_hat": float(np.mean(sigma2_values) * batch_size * sequence_length),
                "rho_last_relative_change": _relative_last_change(all_rho),
                "sigma2_last_relative_change": _relative_last_change(all_sigma2),
            }
        )

        selected = events.copy()
        selected["run_id"] = str(run_id)
        selected["batch_size"] = batch_size
        selected["sequence_length"] = sequence_length
        event_rows.append(selected)

    return pd.DataFrame(summary_rows), pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()


def _fit_target(per_run: pd.DataFrame, target: str, robust_delta: float) -> dict:
    data = per_run.dropna(subset=["batch_size", target])
    data = data[(data["batch_size"] > 0) & (data[target] > 0)]
    if len(data) < 4 or data["batch_size"].nunique() < 4:
        raise RuntimeError(
            f"Need at least four distinct batch sizes to fit {target}; got "
            f"{data['batch_size'].nunique()}."
        )

    X = data[["batch_size"]].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    shifted = _fit_shifted_power_law(X, y, ["batch_size"], robust_delta)

    # Also report an unshifted log-log fit.  It is less flexible and therefore
    # a useful diagnostic when the shifted fit is weakly identified.
    exponent, log_c = np.polyfit(np.log(X[:, 0]), np.log(y), deg=1)
    prediction = np.exp(log_c) * X[:, 0] ** exponent
    unshifted = {
        "c": float(np.exp(log_c)),
        "exponent": float(exponent),
        "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "mape_percent": float(np.mean(np.abs((prediction - y) / y)) * 100.0),
    }
    return {"shifted_power_law": shifted, "unshifted_power_law": unshifted}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    per_run, events = aggregate_runs(df, args)
    if per_run.empty:
        raise RuntimeError("No runs contained enough distinct rho/sigma trace events.")

    fits = {
        "rho": _fit_target(per_run, "rho_hat", args.robust_delta),
        "sigma2": _fit_target(per_run, "sigma2_hat", args.robust_delta),
    }
    equations = {
        name: _equation_string(name, result["shifted_power_law"])
        for name, result in fits.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output_dir / "per_run_batch_noise.csv", index=False)
    events.to_csv(args.output_dir / "noise_events.csv", index=False)
    payload = {
        "input_csv": str(args.input_csv),
        "tail_events": args.tail_events,
        "num_runs": int(len(per_run)),
        "num_batch_sizes": int(per_run["batch_size"].nunique()),
        "fits": fits,
        "equations": equations,
    }
    with (args.output_dir / "batch_noise_fit.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(per_run.to_string(index=False))
    print("\nFitted equations:")
    for equation in equations.values():
        print(equation)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
