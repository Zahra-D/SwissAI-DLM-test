#!/usr/bin/env python3
"""Plot final validation NLL versus SCION FW stepsize across token budgets."""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from plot_scion_extending_budget_lr import (
    BASE_LR,
    predicted_lr,
    run_dimensions,
    run_lr,
    run_seed,
)


def final_val_nll(run):
  """Return the final finite val/nll and its logged step."""
  summary_value = run.summary.get("val/nll")
  summary_step = run.summary.get("trainer/global_step", run.summary.get("global_step", 0))
  if summary_value is not None and math.isfinite(float(summary_value)):
    return float(summary_value), int(summary_step or 0), "summary"

  last_value = None
  last_step = 0
  # Some legacy runs return no rows when a nonexistent step key is requested
  # together with the metric, so request only the metric in the fallback.
  for row in run.scan_history(keys=["val/nll"]):
    value = row.get("val/nll")
    if value is None or not math.isfinite(float(value)):
      continue
    last_value = float(value)
    last_step = int(row.get("trainer/global_step") or row.get("_step") or last_step)
  if last_value is None:
    raise ValueError("no finite val/nll in summary or history")
  return last_value, last_step, "history"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "SwissAI_DLM"))
  parser.add_argument("--project", default=os.environ.get(
      "WANDB_PROJECT", "SwissAI_Extending_Budget"))
  parser.add_argument("--group", default=os.environ.get(
      "WANDB_GROUP", "scion_base_model_token_budget_extension_lr_grid_L12H768_S2048"))
  parser.add_argument("--seed", default="all",
                      help="integer seed to plot, or 'all' to average seeds")
  parser.add_argument("--require-seeds", type=int, default=1,
                      help="omit points with fewer completed seeds")
  parser.add_argument("--output-dir", default=os.environ.get(
      "PLOT_OUTPUT_DIR", "wandb_reports/extending_budget_val_nll"))
  parser.add_argument("--no-upload", action="store_true")
  args = parser.parse_args()

  api = wandb.Api(timeout=120)
  runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})
  selected_seed = None if args.seed.lower() == "all" else int(args.seed)
  records = []
  skipped = []
  for run in runs:
    if run.state != "finished":
      skipped.append((run.name, f"state={run.state}"))
      continue
    try:
      mult, gbs = run_dimensions(run)
      seed = run_seed(run)
      if selected_seed is not None and seed != selected_seed:
        continue
      value, final_step, source = final_val_nll(run)
      records.append({
          "budget_multiplier": mult,
          "tpp": 20 * mult,
          "gbs": gbs,
          "lr": run_lr(run),
          "theory_lr": predicted_lr(mult, gbs, include_sigma_star=False),
          "theory_lr_sigma_star": predicted_lr(mult, gbs, include_sigma_star=True),
          "seed": seed,
          "final_val_nll": value,
          "final_global_step": final_step,
          "value_source": source,
          "run_name": run.name,
          "run_id": run.id,
          "run_url": run.url,
      })
    except Exception as error:
      skipped.append((run.name, str(error)))

  if not records:
    raise SystemExit("No finished runs with validation NLL were found.")

  # Retries may create duplicate runs for one (budget, LR, seed). Keep the run
  # that progressed furthest so a retry is never counted as another seed.
  unique = {}
  for record in records:
    key = (record["budget_multiplier"], record["lr"], record["seed"])
    previous = unique.get(key)
    if previous is None or record["final_global_step"] > previous["final_global_step"]:
      unique[key] = record
  records = list(unique.values())

  seed_counts = defaultdict(set)
  for record in records:
    seed_counts[(record["budget_multiplier"], record["lr"])].add(record["seed"])
  records = [record for record in records
             if len(seed_counts[(record["budget_multiplier"], record["lr"])])
             >= args.require_seeds]
  if not records:
    raise SystemExit(f"No points have {args.require_seeds} completed seed(s).")

  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  per_seed_path = output_dir / "figure5_lr_final_val_nll.csv"
  fieldnames = list(records[0])
  with per_seed_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sorted(records, key=lambda r: (r["budget_multiplier"], r["lr"], r["seed"])))

  groups = defaultdict(list)
  for record in records:
    groups[(record["budget_multiplier"], record["lr"])].append(record)
  averaged = []
  for (mult, lr), rows in sorted(groups.items()):
    values = [row["final_val_nll"] for row in rows]
    averaged.append({
        "budget_multiplier": mult,
        "tpp": rows[0]["tpp"],
        "gbs": rows[0]["gbs"],
        "lr": lr,
        "theory_lr": rows[0]["theory_lr"],
        "theory_lr_sigma_star": rows[0]["theory_lr_sigma_star"],
        "num_seeds": len(rows),
        "seeds": ";".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
        "mean_final_val_nll": float(np.mean(values)),
        "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    })
  averaged_path = output_dir / "figure5_lr_final_val_nll_averaged.csv"
  averaged_fields = list(averaged[0])
  with averaged_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=averaged_fields)
    writer.writeheader()
    writer.writerows(averaged)

  by_budget = defaultdict(list)
  for row in averaged:
    by_budget[row["budget_multiplier"]].append(row)
  budgets = sorted(by_budget)
  fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 4.2), squeeze=False)
  for axis, mult in zip(axes[0], budgets):
    rows = sorted(by_budget[mult], key=lambda row: row["lr"])
    xs = np.array([row["lr"] for row in rows])
    ys = np.array([row["mean_final_val_nll"] for row in rows])
    yerr = np.array([row["std_across_seeds"] for row in rows])
    axis.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3,
                  color="#2774AE", label="observed")
    theory_lr = rows[0]["theory_lr"]
    sigma_lr = rows[0]["theory_lr_sigma_star"]
    empirical_lr = float(xs[int(np.argmin(ys))])
    axis.axvline(theory_lr, color="#D55E00", linestyle="--", linewidth=1.8,
                 label=f"BST theory {theory_lr:.5f}")
    axis.axvline(sigma_lr, color="#CC79A7", linestyle="-.", linewidth=1.5,
                 label=f"with σ* {sigma_lr:.5f}")
    axis.axvline(BASE_LR, color="#555555", linestyle=":", linewidth=1.6,
                 label="base β 0.0225")
    axis.scatter([empirical_lr], [float(np.min(ys))], marker="*", s=150,
                 color="#009E73", zorder=5, label=f"best {empirical_lr:.5f}")
    axis.set_title(f"{20 * mult} TPP (B={rows[0]['gbs']})")
    axis.set_xlabel("Frank–Wolfe stepsize β")
    axis.grid(alpha=0.25)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2))
    axis.legend(fontsize=8)
  axes[0][0].set_ylabel("final validation NLL")
  available_seeds = sorted({record["seed"] for record in records})
  seed_title = ("average of available seeds " + ", ".join(map(str, available_seeds))
                if selected_seed is None else f"seed {selected_seed}")
  fig.suptitle(f"Extending-budget SCION validation NLL ({seed_title})", fontsize=14)
  fig.tight_layout()
  png_path = output_dir / "figure5_lr_final_val_nll.png"
  pdf_path = output_dir / "figure5_lr_final_val_nll.pdf"
  fig.savefig(png_path, dpi=220, bbox_inches="tight")
  fig.savefig(pdf_path, bbox_inches="tight")
  plt.close(fig)

  if not args.no_upload:
    analysis = wandb.init(
        entity=args.entity, project=args.project, group="analysis", job_type="analysis",
        name=f"figure5_final_val_nll_seed{args.seed}_minseeds{args.require_seeds}",
        config={
            "source_group": args.group,
            "seed_selection": args.seed,
            "minimum_completed_seeds_per_point": args.require_seeds,
            "metric": "val/nll",
            "statistic": "final validation NLL per run, then mean across seeds",
            "moving_average": False,
        },
        tags=["extending_budget", "fig5_lr_sweep", "validation_nll", "analysis"],
    )
    table = wandb.Table(columns=fieldnames)
    for record in sorted(records, key=lambda r: (r["budget_multiplier"], r["lr"], r["seed"])):
      table.add_data(*(record[name] for name in fieldnames))
    averaged_table = wandb.Table(columns=averaged_fields)
    for record in averaged:
      averaged_table.add_data(*(record[name] for name in averaged_fields))
    analysis.log({"figure5/final_val_nll": wandb.Image(str(png_path)),
                  "figure5/val_nll_results": table,
                  "figure5/val_nll_averaged_results": averaged_table})
    artifact = wandb.Artifact("scion-extending-budget-figure5-val-nll", type="analysis")
    for path in (per_seed_path, averaged_path, png_path, pdf_path):
      artifact.add_file(str(path))
    analysis.log_artifact(artifact)
    analysis.finish()

  print(f"Plotted {len(records)} runs and {len(averaged)} LR points across {len(budgets)} budgets.")
  print(f"PNG: {png_path}")
  print(f"PDF: {pdf_path}")
  print(f"Per-seed CSV: {per_seed_path}")
  print(f"Averaged CSV: {averaged_path}")
  if skipped:
    print(f"Skipped {len(skipped)} non-finished or unusable runs.")


if __name__ == "__main__":
  main()
