#!/usr/bin/env python3
"""Report early train/validation metrics for transferred-radii W&B runs."""

import argparse
import math
import re

import wandb


TARGET_RADII = (70.0, 1.8, 0.136, 0.4375)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_transferred_radii(run):
    config = run.config
    optim = config.get("optim", {})
    if isinstance(optim, dict):
        radii = tuple(number(optim.get(key)) for key in (
            "scale_embed", "scale_bias", "scale_layer_norm", "scale_matrix"
        ))
        if all(value is not None for value in radii) and all(
            math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-10)
            for value, expected in zip(radii, TARGET_RADII)
        ):
            return True
    candidates = [
        ("scale_embed", "scale_bias", "scale_layer_norm", "scale_matrix"),
        ("optim.scale_embed", "optim.scale_bias", "optim.scale_layer_norm", "optim.scale_matrix"),
    ]
    for keys in candidates:
        radii = tuple(number(config.get(key)) for key in keys)
        if all(value is not None for value in radii) and all(
            math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-10)
            for value, expected in zip(radii, TARGET_RADII)
        ):
            return True
    # Older runs did not always persist Hydra's nested `optim` config in a
    # queryable form.  Their run/group names carry the explicit variant.
    return "transferred" in f"{run.name or ''} {run.group or ''}".lower()


def closest_values(run, target):
    result = {
        "train_step": None, "train_elbo": None,
        "val_step": None, "val_nll": None,
    }
    # Do not pass `keys=` here: W&B then returns only rows that contain *all*
    # requested keys, while validation and training scalars are logged in
    # separate records.
    for row in run.scan_history(page_size=1000):
        step = number(row.get("trainer/global_step"))
        if step is None or step > target:
            continue
        step = int(step)
        if row.get("train/elbo") is not None and (result["train_step"] is None or step >= result["train_step"]):
            result["train_step"] = step
            result["train_elbo"] = number(row["train/elbo"])
        if row.get("val/nll") is not None and (result["val_step"] is None or step >= result["val_step"]):
            result["val_step"] = step
            result["val_nll"] = number(row["val/nll"])
    return result


def config_value(config, *keys):
    for key in keys:
        if key in config:
            return config[key]
    return None


def fmt(value):
    return "—" if value is None else f"{value:.6f}"


def hp_from_name(name, field):
    if field == "beta":
        match = re.search(r"beta([0-9]+)em([0-9]+)", name)
        if match:
            return f"{match.group(1)}e-{match.group(2)}"
    if field == "momentum":
        match = re.search(r"alpha([0-9]+)p([0-9]+)", name)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="SwissAI_DLM/SwissAI_Scion_B3_HP_Tuning")
    parser.add_argument("--step", type=int, default=1000)
    args = parser.parse_args()

    runs = wandb.Api().runs(args.project, per_page=100)
    rows = []
    for run in runs:
        if "baseline" in run.tags or run.job_type == "baseline_import":
            continue
        if not has_transferred_radii(run):
            continue
        metrics = closest_values(run, args.step)
        # Keep runs that have reached either relevant metric.
        if metrics["train_step"] is None and metrics["val_step"] is None:
            continue
        config = run.config
        optim = config.get("optim", {}) if isinstance(config.get("optim", {}), dict) else {}
        rows.append({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "beta": config_value(optim, "lr") or config_value(config, "optim.lr", "lr") or hp_from_name(run.name, "beta"),
            "momentum": config_value(optim, "momentum", "momentum_alpha") or config_value(config, "optim.momentum", "momentum", "optim.momentum_alpha") or hp_from_name(run.name, "momentum"),
            **metrics,
            "url": run.url,
        })

    rows.sort(key=lambda row: (
        float("inf") if number(row["beta"]) is None else number(row["beta"]),
        float("inf") if number(row["momentum"]) is None else number(row["momentum"]),
        row["name"],
    ))
    print(f"Transferred radii: embed={TARGET_RADII[0]}, bias={TARGET_RADII[1]}, norm={TARGET_RADII[2]}, matrix={TARGET_RADII[3]}")
    print(f"Target step: {args.step}; each metric uses its latest logged value at or before target.")
    print("| beta | momentum | train step | train ELBO | val step | val NLL | state | run |")
    print("|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        beta = "—" if row["beta"] is None else str(row["beta"])
        momentum = "—" if row["momentum"] is None else str(row["momentum"])
        print(
            f"| {beta} | {momentum} | {row['train_step'] or '—'} | {fmt(row['train_elbo'])} "
            f"| {row['val_step'] or '—'} | {fmt(row['val_nll'])} | {row['state']} "
            f"| [{row['id']}]({row['url']}) |"
        )
    print(f"runs_reported={len(rows)}")


if __name__ == "__main__":
    main()
