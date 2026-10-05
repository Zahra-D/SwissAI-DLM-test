#!/usr/bin/env python3
"""Copy an existing W&B run's config, summary, and scalar history as a baseline.

The source run is read through the W&B public API.  A new, independent run is
created in the destination project; the source is never modified.
"""

import argparse
import json
from collections import defaultdict

import wandb


def json_safe(value):
    """Return a value suitable for W&B config/history, or None if unsupported."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="entity/project/run_id of the existing run")
    parser.add_argument("--entity", default="SwissAI_DLM")
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    source = wandb.Api().run(args.source)
    config = {key: json_safe(value) for key, value in source.config.items()}
    config.update({
        "baseline/source_run": source.path,
        "baseline/source_url": source.url,
        "baseline/copied_for_comparison": True,
    })

    tags = sorted(set(list(source.tags) + ["baseline", "imported-history"]))
    dest = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        name=args.name,
        job_type="baseline_import",
        tags=tags,
        config=config,
        notes=("Read-only copy of historical baseline: " + source.url),
    )

    # The source may contain several records for a single training step (e.g.
    # a performance record then a validation record), and its API does not
    # guarantee these records arrive in monotonically increasing global-step
    # order.  Merge per training step first: this preserves every scalar while
    # satisfying W&B's monotonic `step` requirement in the destination.
    history_by_step = defaultdict(dict)
    for row in source.scan_history(page_size=1000):
        clean = {
            key: json_safe(value)
            for key, value in row.items()
            if not key.startswith("_") and json_safe(value) is not None
        }
        step = clean.get("trainer/global_step")
        if not clean or not isinstance(step, (int, float)):
            continue
        history_by_step[int(step)].update(clean)

    for step in sorted(history_by_step):
        wandb.log(history_by_step[step], step=step)
    history_rows = len(history_by_step)

    summary = {
        key: json_safe(value)
        for key, value in source.summary.items()
        if not key.startswith("_") and json_safe(value) is not None
    }
    dest.summary.update(summary)
    dest.summary["baseline/history_rows_copied"] = history_rows
    dest.summary["baseline/source_run"] = source.path
    dest.summary["baseline/source_url"] = source.url
    dest.finish()
    print(f"source={source.path}")
    print(f"destination={args.entity}/{args.project}/{dest.id}")
    print(f"url={dest.url}")
    print(f"history_rows_copied={history_rows}")


if __name__ == "__main__":
    main()
