#!/usr/bin/env python3
"""
Estimate Scion constants and fit shifted power-law functions.

This script follows the Section 6 workflow in arXiv:2603.21191:
- L (Sec. 6.4): average local curvature proxy over the last N training points.
- mu (Sec. 6.2 + 6.4): robust linear fit (Huber loss) between dual gradient norm and training loss.
- rho (Sec. 6.4): average ratio proxy for ||g - grad f||_* / ||g - grad f||_2 over last N points.

Expected input: one CSV with step-level logs across runs.
Each row should contain a run identifier, model-size metadata, and metric columns.

ScionTrace metric mapping used by default:
- L   <- `stats/local_smooth_spec` (fallback: `stats/num_nuc` / `stats/den_spec`)
- mu  <- Huber slope of `stats/grad_norm_nuc_power_1` vs `trainer/loss`
- rho <- `rho/rho_over_averaged_norms` (optional; fallback: `rho/averaged_rho_over_samples`)
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor


@dataclass
class MetricSpec:
    name: str
    source_col: str | None = None


def _safe_mean_last(values: pd.Series, tail: int) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    if tail > 0 and vals.size > tail:
        vals = vals[-tail:]
    return float(np.mean(vals))


def _fit_mu_huber(
    group: pd.DataFrame,
    loss_col: str,
    dual_grad_col: str,
    loss_max_for_fit: float | None,
    min_points: int,
) -> float:
    x = pd.to_numeric(group[loss_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group[dual_grad_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if loss_max_for_fit is not None:
        mask &= x <= loss_max_for_fit

    x = x[mask]
    y = y[mask]

    if x.size < min_points:
        return float("nan")

    X = x.reshape(-1, 1)
    # y ~= mu * loss + b
    model = HuberRegressor(fit_intercept=True)
    model.fit(X, y)
    return float(model.coef_[0])


def estimate_run_constants(
    df: pd.DataFrame,
    run_col: str,
    n_layer_col: str,
    n_embd_col: str,
    batch_col: str,
    l_proxy_col: str,
    rho_proxy_col: str,
    train_loss_col: str,
    dual_grad_col: str,
    tail: int,
    mu_loss_max: float | None,
    min_points_mu: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for run_id, group in df.groupby(run_col, sort=False):
        g = group.sort_values("step") if "step" in group.columns else group

        n_layer = pd.to_numeric(g[n_layer_col], errors="coerce").dropna()
        n_embd = pd.to_numeric(g[n_embd_col], errors="coerce").dropna()
        bsize = pd.to_numeric(g[batch_col], errors="coerce").dropna()

        if n_layer.empty or n_embd.empty or bsize.empty:
            continue

        if l_proxy_col in g.columns:
            L_hat = _safe_mean_last(g[l_proxy_col], tail)
        elif "stats/num_nuc" in g.columns and "stats/den_spec" in g.columns:
            num = pd.to_numeric(g["stats/num_nuc"], errors="coerce")
            den = pd.to_numeric(g["stats/den_spec"], errors="coerce").replace(0.0, np.nan)
            L_hat = _safe_mean_last(num / den, tail)
        else:
            L_hat = float("nan")

        if rho_proxy_col in g.columns:
            rho_hat = _safe_mean_last(g[rho_proxy_col], tail)
        elif "rho/averaged_rho_over_samples" in g.columns:
            rho_hat = _safe_mean_last(g["rho/averaged_rho_over_samples"], tail)
        else:
            rho_hat = float("nan")
        mu_hat = _fit_mu_huber(
            group=g,
            loss_col=train_loss_col,
            dual_grad_col=dual_grad_col,
            loss_max_for_fit=mu_loss_max,
            min_points=min_points_mu,
        )

        rows.append(
            {
                "run_id": str(run_id),
                "n_layer": float(n_layer.iloc[-1]),
                "n_embd": float(n_embd.iloc[-1]),
                "batch_size": float(bsize.iloc[-1]),
                "L_hat": L_hat,
                "mu_hat": mu_hat,
                "rho_hat": rho_hat,
                "L_source": l_proxy_col if l_proxy_col in g.columns else ("stats/num_nuc_over_den_spec" if ("stats/num_nuc" in g.columns and "stats/den_spec" in g.columns) else "missing"),
                "rho_source": rho_proxy_col if rho_proxy_col in g.columns else ("rho/averaged_rho_over_samples" if "rho/averaged_rho_over_samples" in g.columns else "missing"),
                "num_points": float(len(g)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.dropna(subset=["n_layer", "n_embd", "batch_size", "L_hat", "mu_hat"])


def _powerlaw_predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params = [log_c, p1..pd, s1..sd], y = exp(log_c) * prod((x_i - s_i)^p_i)
    d = X.shape[1]
    log_c = params[0]
    p = params[1 : 1 + d]
    s = params[1 + d : 1 + 2 * d]

    shifted = X - s
    if np.any(shifted <= 0):
        # caller handles penalty through objective, but keep predict numerically safe
        return np.full((X.shape[0],), np.nan)

    return np.exp(log_c) * np.exp(np.sum(p * np.log(shifted), axis=1))


def _fit_shifted_power_law(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    robust_delta: float,
) -> Dict[str, object]:
    d = X.shape[1]

    y = y.astype(float)
    X = X.astype(float)

    if np.any(y <= 0):
        raise ValueError("All target values must be positive for shifted power-law fitting.")

    # Init: zero shifts, log-linear fit for c and exponents.
    z = np.log(X)
    A = np.concatenate([np.ones((X.shape[0], 1)), z], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    log_c0 = coef[0]
    p0 = coef[1:]
    s0 = np.zeros(d)
    theta0 = np.concatenate([[log_c0], p0, s0])

    x_min = X.min(axis=0)

    def objective(theta: np.ndarray) -> float:
        log_c = theta[0]
        p = theta[1 : 1 + d]
        s = theta[1 + d : 1 + 2 * d]

        shifted = X - s
        if np.any(shifted <= 1e-12):
            return 1e12

        pred_log = log_c + np.sum(p * np.log(shifted), axis=1)
        resid = pred_log - np.log(y)

        abs_r = np.abs(resid)
        quad = abs_r <= robust_delta
        huber = np.where(quad, 0.5 * resid * resid, robust_delta * (abs_r - 0.5 * robust_delta))

        # Soft regularization on shift magnitude for stability.
        reg = 1e-4 * np.sum((s / np.maximum(x_min, 1.0)) ** 2)
        return float(np.mean(huber) + reg)

    bounds = []
    bounds.append((None, None))
    for _ in range(d):
        bounds.append((None, None))
    for j in range(d):
        # Keep shifts smaller than minimum observed x_j to preserve positivity.
        bounds.append((None, float(x_min[j] - 1e-6)))

    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)

    theta = res.x
    y_pred = _powerlaw_predict(X, theta)
    rmse = float(np.sqrt(np.nanmean((y_pred - y) ** 2)))
    mape = float(np.nanmean(np.abs((y_pred - y) / np.maximum(y, 1e-12))) * 100.0)

    log_c = float(theta[0])
    p = theta[1 : 1 + d]
    s = theta[1 + d : 1 + 2 * d]

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "objective": float(res.fun),
        "rmse": rmse,
        "mape_percent": mape,
        "coef": {
            "c": float(math.exp(log_c)),
            "exponents": {k: float(v) for k, v in zip(feature_names, p)},
            "shifts": {k: float(v) for k, v in zip(feature_names, s)},
        },
    }


def _equation_string(name: str, fit: Dict[str, object]) -> str:
    c = fit["coef"]["c"]
    exponents = fit["coef"]["exponents"]
    shifts = fit["coef"]["shifts"]

    terms = [f"{c:.6g}"]
    for feat in exponents.keys():
        p = exponents[feat]
        s = shifts[feat]
        sign = "-" if s >= 0 else "+"
        terms.append(f"({feat} {sign} {abs(s):.6g})^{p:.6g}")
    return f"{name} = " + " * ".join(terms)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit L, mu, rho scaling laws from run logs.")

    p.add_argument("--input_csv", type=Path, required=True, help="Step-level metrics CSV across runs.")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/analysis/constant_fits"))

    # Column mapping
    p.add_argument("--run_col", default="run_id")
    p.add_argument("--step_col", default="step")
    p.add_argument("--n_layer_col", default="n_layer")
    p.add_argument("--n_embd_col", default="n_embd")
    p.add_argument("--batch_col", default="batch_size")

    p.add_argument("--train_loss_col", default="trainer/loss")
    p.add_argument("--dual_grad_col", default="stats/grad_norm_nuc_power_1")

    p.add_argument("--l_proxy_col", default="stats/local_smooth_spec")
    p.add_argument("--rho_proxy_col", default="rho/rho_over_averaged_norms")
    p.add_argument("--constants", nargs="+", choices=["mu", "L", "rho"], default=["mu", "L", "rho"])

    p.add_argument("--tail", type=int, default=100, help="Last N points for L/rho averaging.")
    p.add_argument("--mu_loss_max", type=float, default=5.0, help="Use loss <= this value for mu fit (Sec. 6.2).")
    p.add_argument("--min_points_mu", type=int, default=20)

    # Shifted power-law fit settings
    p.add_argument("--robust_delta", type=float, default=0.15)

    # Feature sets per constant
    p.add_argument("--mu_features", nargs="+", default=["n_layer", "n_embd"])
    p.add_argument("--L_features", nargs="+", default=["n_layer", "n_embd"])
    p.add_argument("--rho_features", nargs="+", default=["n_layer", "n_embd", "batch_size"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    # Normalize metadata aliases if present.
    alias_map = {
        "config/n_layer": args.n_layer_col,
        "config.n_layer": args.n_layer_col,
        "model/n_layer": args.n_layer_col,
        "config/n_embd": args.n_embd_col,
        "config.n_embd": args.n_embd_col,
        "model/n_embd": args.n_embd_col,
        "config/batch_size": args.batch_col,
        "config.batch_size": args.batch_col,
        "trainer/global_step": args.step_col,
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    required = [
        args.run_col,
        args.n_layer_col,
        args.n_embd_col,
        args.batch_col,
        args.train_loss_col,
        args.dual_grad_col,
    ]
    if args.step_col in df.columns and args.step_col != "step":
        df["step"] = df[args.step_col]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    per_run = estimate_run_constants(
        df=df,
        run_col=args.run_col,
        n_layer_col=args.n_layer_col,
        n_embd_col=args.n_embd_col,
        batch_col=args.batch_col,
        l_proxy_col=args.l_proxy_col,
        rho_proxy_col=args.rho_proxy_col,
        train_loss_col=args.train_loss_col,
        dual_grad_col=args.dual_grad_col,
        tail=args.tail,
        mu_loss_max=args.mu_loss_max,
        min_points_mu=args.min_points_mu,
    )

    if per_run.empty:
        raise RuntimeError("No valid runs found after filtering and aggregation.")

    # Fit scaling laws
    fit_results: Dict[str, Dict[str, object]] = {}

    def fit_one(target_col: str, features: Sequence[str]) -> Dict[str, object]:
        data = per_run.dropna(subset=[target_col, *features])
        if len(data) < max(6, len(features) + 2):
            raise RuntimeError(
                f"Not enough runs to fit {target_col} with features {features}. "
                f"Need at least {max(6, len(features) + 2)}, got {len(data)}."
            )
        X = data[list(features)].to_numpy(dtype=float)
        y = data[target_col].to_numpy(dtype=float)
        return _fit_shifted_power_law(X, y, features, robust_delta=args.robust_delta)

    if "mu" in args.constants:
        fit_results["mu"] = fit_one("mu_hat", args.mu_features)
    if "L" in args.constants:
        fit_results["L"] = fit_one("L_hat", args.L_features)
    if "rho" in args.constants:
        fit_results["rho"] = fit_one("rho_hat", args.rho_features)

    equations = {
        name: _equation_string(name, fit)
        for name, fit in fit_results.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = args.output_dir / "per_run_constants.csv"
    per_run.to_csv(per_run_path, index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "num_runs": int(len(per_run)),
        "settings": {
            "tail": args.tail,
            "mu_loss_max": args.mu_loss_max,
            "min_points_mu": args.min_points_mu,
            "robust_delta": args.robust_delta,
            "constants": args.constants,
            "mu_features": args.mu_features,
            "L_features": args.L_features,
            "rho_features": args.rho_features,
        },
        "fits": fit_results,
        "equations": equations,
    }

    summary_path = args.output_dir / "fit_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- per-run constants: {per_run_path}")
    print(f"- fit summary:       {summary_path}")
    print("\nFitted equations:")
    print(equations["mu"])
    print(equations["L"])
    print(equations["rho"])


if __name__ == "__main__":
    main()
