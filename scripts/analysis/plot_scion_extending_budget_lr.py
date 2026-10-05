#!/usr/bin/env python3
"""Create and upload a Figure-5-style final-loss-vs-LR plot from W&B."""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


TAG_PATTERNS = {
    "mult": re.compile(r"^budget_mult_(\d+)$"),
    "gbs": re.compile(r"^gbs_(\d+)$"),
    "theory_lr": re.compile(r"^theory_lr_([0-9.eE+-]+)$"),
    "kind": re.compile(r"^lr_kind_(.+)$"),
}

RHO_EXPONENT = -0.00291078
SIGMA_STAR_EXPONENT = 0.2005635
BASE_BATCH = 256
BASE_LR = 0.0225


def tag_value(tags, key, cast):
  pattern = TAG_PATTERNS[key]
  for tag in tags:
    match = pattern.match(tag)
    if match:
      return cast(match.group(1))
  raise ValueError(f"missing {key!r} tag")


def run_dimensions(run):
  """Read the new structured tags, or parse the completed legacy grid name."""
  try:
    return tag_value(run.tags, "mult", int), tag_value(run.tags, "gbs", int)
  except ValueError:
    match = re.search(r"_(\d+)x_gbs(\d+)_lr", run.name)
    if not match:
      raise ValueError("could not determine budget multiplier and GBS")
    return int(match.group(1)), int(match.group(2))


def run_seed(run):
  value = run.config.get("seed", None)
  if value is not None:
    return int(value)
  match = re.search(r"_seed(\d+)(?:_|$)", run.name)
  return int(match.group(1)) if match else 4


def predicted_lr(mult, gbs, include_sigma_star=False):
  ratio = math.sqrt(1.0 / mult) * (gbs / BASE_BATCH) ** RHO_EXPONENT
  if include_sigma_star:
    ratio *= (gbs / BASE_BATCH) ** SIGMA_STAR_EXPONENT
  return BASE_LR * ratio ** (2.0 / 3.0)


def nested(config, *keys):
  value = config
  for key in keys:
    if not isinstance(value, dict) or key not in value:
      return None
    value = value[key]
  return value


def run_lr(run):
  value = nested(run.config, "optim", "lr")
  if value is None:
    value = run.config.get("optim.lr")
  if value is None:
    match = re.search(r"_lr([0-9]+p[0-9]+)_", run.name)
    if match:
      value = match.group(1).replace("p", ".")
  if value is None:
    raise ValueError("could not determine optim.lr")
  return float(value)


def loss_history(run):
  values = []
  # Requesting a nonexistent step key together with trainer/loss makes the W&B
  # history API return no rows for these legacy runs. The logging cadence is
  # stored in the run config, so trainer/loss alone is sufficient.
  for row in run.scan_history(keys=["trainer/loss"]):
    value = row.get("trainer/loss")
    if value is not None and math.isfinite(float(value)):
      values.append(float(value))
  return values


def log_interval(run):
  value = nested(run.config, "trainer", "log_every_n_steps")
  if value is None:
    value = run.config.get("trainer.log_every_n_steps")
  return int(value) if value is not None else 25


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "SwissAI_DLM"))
  parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "SwissAI_Extending_Budget"))
  parser.add_argument("--group", default=os.environ.get(
      "WANDB_GROUP", "scion_base_model_token_budget_extension_lr_grid_L12H768_S2048"))
  parser.add_argument("--seed", default="4",
                      help="integer seed to plot, or 'all' to average seeds")
  parser.add_argument("--require-seeds", type=int, default=1,
                      help="omit (budget, LR) points with fewer completed seeds")
  parser.add_argument("--window-steps", type=int, default=500)
  parser.add_argument("--output-dir", default=os.environ.get(
      "PLOT_OUTPUT_DIR", "wandb_reports/extending_budget_lr_fig5"))
  parser.add_argument("--include-running", action="store_true")
  parser.add_argument("--no-upload", action="store_true")
  args = parser.parse_args()

  api = wandb.Api(timeout=120)
  runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})
  states = {"finished"}
  if args.include_running:
    states.add("running")

  records = []
  skipped = []
  selected_seed = None if args.seed.lower() == "all" else int(args.seed)
  for run in runs:
    if run.state not in states:
      skipped.append((run.name, f"state={run.state}"))
      continue
    try:
      mult, gbs = run_dimensions(run)
      seed = run_seed(run)
      if selected_seed is not None and seed != selected_seed:
        continue
      theory_lr = predicted_lr(mult, gbs, include_sigma_star=False)
      theory_lr_sigma_star = predicted_lr(mult, gbs, include_sigma_star=True)
      lr = run_lr(run)
      losses = loss_history(run)
      if not losses:
        skipped.append((run.name, "no trainer/loss history"))
        continue
      interval = log_interval(run)
      window_observations = max(1, math.ceil(args.window_steps / interval))
      final_window = losses[-window_observations:]
      final_step = run.summary.get("trainer/global_step")
      if final_step is None:
        final_step = run.summary.get("global_step")
      if final_step is None:
        final_step = len(losses) * interval
      final_step = int(final_step)
      records.append({
          "budget_multiplier": mult,
          "tpp": 20 * mult,
          "gbs": gbs,
          "lr": lr,
          "theory_lr": theory_lr,
          "theory_lr_sigma_star": theory_lr_sigma_star,
          "seed": seed,
          "final_loss_mean_500": float(np.mean(final_window)),
          "final_loss_std_500": float(np.std(final_window, ddof=1)),
          "window_observations": len(final_window),
          "wandb_log_interval_steps": interval,
          "final_global_step": final_step,
          "run_name": run.name,
          "run_id": run.id,
          "run_url": run.url,
      })
    except Exception as error:  # Keep one malformed run from blocking the plot.
      skipped.append((run.name, str(error)))

  if not records:
    details = "\n".join(f"  {name}: {reason}" for name, reason in skipped)
    raise SystemExit(f"No plottable runs found.\n{details}")

  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  # Retries can leave multiple finished W&B runs for the same setup. Keep the
  # most complete one rather than counting a retry as another seed.
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
    raise SystemExit(
        f"No points have the required {args.require_seeds} completed seed(s).")

  csv_path = output_dir / "figure5_lr_final_loss_last500steps.csv"
  fieldnames = list(records[0])
  with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sorted(records, key=lambda row: (row["budget_multiplier"], row["lr"])))

  averaged_records = []
  aggregate_groups = defaultdict(list)
  for record in records:
    aggregate_groups[(record["budget_multiplier"], record["lr"])].append(record)
  for (mult, lr), rows in sorted(aggregate_groups.items()):
    losses = [row["final_loss_mean_500"] for row in rows]
    averaged_records.append({
        "budget_multiplier": mult,
        "tpp": rows[0]["tpp"],
        "gbs": rows[0]["gbs"],
        "lr": lr,
        "theory_lr": rows[0]["theory_lr"],
        "theory_lr_sigma_star": rows[0]["theory_lr_sigma_star"],
        "num_seeds": len(rows),
        "seeds": ";".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
        "mean_final_loss": float(np.mean(losses)),
        "std_across_seeds": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
    })
  averaged_csv_path = output_dir / "figure5_lr_final_loss_last500steps_averaged.csv"
  averaged_fieldnames = list(averaged_records[0])
  with averaged_csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=averaged_fieldnames)
    writer.writeheader()
    writer.writerows(averaged_records)

  by_budget = defaultdict(list)
  for record in records:
    by_budget[record["budget_multiplier"]].append(record)
  budgets = sorted(by_budget)
  fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 4.2), squeeze=False)
  axes = axes[0]
  for axis, mult in zip(axes, budgets):
    rows = sorted(by_budget[mult], key=lambda row: row["lr"])
    # If more seeds are added later, average seed-level final-window means at
    # each LR. With one seed this is the identity operation requested here.
    grouped = defaultdict(list)
    for row in rows:
      grouped[row["lr"]].append(row["final_loss_mean_500"])
    xs = np.array(sorted(grouped))
    ys = np.array([np.mean(grouped[x]) for x in xs])
    yerr = np.array([np.std(grouped[x], ddof=1) if len(grouped[x]) > 1 else 0.0 for x in xs])
    axis.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3,
                  color="#2774AE", label="observed")
    theory_lr = rows[0]["theory_lr"]
    empirical_lr = float(xs[int(np.argmin(ys))])
    axis.axvline(theory_lr, color="#D55E00", linestyle="--", linewidth=1.8,
                 label=f"theory {theory_lr:.5f}")
    sigma_star_lr = rows[0]["theory_lr_sigma_star"]
    axis.axvline(sigma_star_lr, color="#CC79A7", linestyle="-.", linewidth=1.5,
                 label=f"with σ* {sigma_star_lr:.5f}")
    axis.axvline(BASE_LR, color="#555555", linestyle=":", linewidth=1.6,
                 label="base LR 0.0225")
    axis.scatter([empirical_lr], [float(np.min(ys))], marker="*", s=150,
                 color="#009E73", zorder=5, label=f"best {empirical_lr:.5f}")
    axis.set_title(f"{20 * mult} TPP (B={rows[0]['gbs']})")
    axis.set_xlabel("Frank–Wolfe stepsize β")
    axis.grid(alpha=0.25)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2))
    axis.legend(fontsize=8)
  axes[0].set_ylabel(f"train loss (W&B observations in final {args.window_steps} steps)")
  available_seeds = sorted({record["seed"] for record in records})
  seed_title = ("average of available seeds "
                + ", ".join(str(seed) for seed in available_seeds)
                if selected_seed is None else f"seed {selected_seed}")
  fig.suptitle(f"Extending-budget SCION learning-rate sweep ({seed_title})", fontsize=14)
  fig.tight_layout()
  png_path = output_dir / "figure5_lr_final_loss_last500steps.png"
  pdf_path = output_dir / "figure5_lr_final_loss_last500steps.pdf"
  fig.savefig(png_path, dpi=220, bbox_inches="tight")
  fig.savefig(pdf_path, bbox_inches="tight")
  plt.close(fig)

  if not args.no_upload:
    analysis = wandb.init(
        entity=args.entity,
        project=args.project,
        group="analysis",
        job_type="analysis",
        name=f"figure5_lr_last{args.window_steps}steps_seed{args.seed}",
        config={
            "source_group": args.group,
            "seed_selection": args.seed,
            "minimum_completed_seeds_per_point": args.require_seeds,
            "window_optimizer_steps": args.window_steps,
            "statistic": "mean of logged W&B losses in final optimizer-step window",
            "limitation": (
                "training logged every 25 steps, so the 500-step window has "
                "about 20 sampled loss observations rather than 500 observations"),
        },
        tags=["extending_budget", "fig5_lr_sweep", "analysis",
              f"last_{args.window_steps}_steps", f"seed_{args.seed}"],
    )
    table = wandb.Table(columns=fieldnames)
    for record in sorted(records, key=lambda row: (row["budget_multiplier"], row["lr"])):
      table.add_data(*(record[name] for name in fieldnames))
    averaged_table = wandb.Table(columns=averaged_fieldnames)
    for record in averaged_records:
      averaged_table.add_data(*(record[name] for name in averaged_fieldnames))
    analysis.log({"figure5/lr_final_loss": wandb.Image(str(png_path)),
                  "figure5/results": table,
                  "figure5/averaged_results": averaged_table})
    artifact = wandb.Artifact("scion-extending-budget-figure5", type="analysis")
    artifact.add_file(str(csv_path))
    artifact.add_file(str(averaged_csv_path))
    artifact.add_file(str(png_path))
    artifact.add_file(str(pdf_path))
    analysis.log_artifact(artifact)
    analysis.finish()

  print(f"Plotted {len(records)} runs across {len(budgets)} budgets.")
  print(f"PNG: {png_path}")
  print(f"PDF: {pdf_path}")
  print(f"CSV: {csv_path}")
  print(f"Averaged CSV: {averaged_csv_path}")
  if skipped:
    print("Skipped runs:")
    for name, reason in skipped:
      print(f"  {name}: {reason}")


if __name__ == "__main__":
  main()
