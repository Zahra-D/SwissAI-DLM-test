#!/usr/bin/env python3
"""Materialize a deterministic shuffled subset of a packed HF dataset.

The expensive source dataset is sampled uniformly without replacement.  The
exact source-row order used by the output is retained as a NumPy array and is
checksummed in a JSON manifest, so the subset can be audited or reconstructed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import datasets
import numpy as np


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--output-cache-dir", type=Path, required=True)
  parser.add_argument("--output-train-name", required=True)
  parser.add_argument("--validation-source", type=Path, required=True)
  parser.add_argument("--validation-name", required=True)
  parser.add_argument("--num-sequences", type=int, required=True)
  parser.add_argument("--sequence-length", type=int, default=2048)
  parser.add_argument("--seed", type=int, default=4)
  parser.add_argument("--max-shard-size", default="1GB")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  started = time.time()
  source = args.source.resolve()
  validation_source = args.validation_source.resolve()
  output_root = args.output_cache_dir.resolve()
  final_dataset = output_root / args.output_train_name
  validation_link = output_root / args.validation_name
  job_tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
  sorted_stage = output_root / f".subset-sorted-{job_tag}"
  final_stage = output_root / f".subset-final-{job_tag}"

  if not source.is_dir() or not (source / "state.json").is_file():
    raise FileNotFoundError(f"Not a load_from_disk dataset: {source}")
  if not validation_source.is_dir():
    raise FileNotFoundError(f"Missing validation dataset: {validation_source}")
  if final_dataset.exists():
    raise FileExistsError(
      f"Refusing to replace existing output dataset: {final_dataset}")
  if sorted_stage.exists() or final_stage.exists():
    raise FileExistsError("A job-specific staging directory already exists")
  if args.num_sequences <= 0 or args.sequence_length <= 0:
    raise ValueError("num-sequences and sequence-length must be positive")

  output_root.mkdir(parents=True, exist_ok=True)
  print(f"Loading packed source metadata: {source}", flush=True)
  source_dataset = datasets.load_from_disk(str(source))
  source_len = len(source_dataset)
  if args.num_sequences > source_len:
    raise ValueError(
      f"Requested {args.num_sequences:,} rows from only {source_len:,}")
  if "input_ids" not in source_dataset.column_names:
    raise ValueError(
      f"Packed source has no input_ids column: {source_dataset.column_names}")

  print(
    f"Selecting {args.num_sequences:,} of {source_len:,} packed rows "
    f"uniformly without replacement (PCG64 seed={args.seed})", flush=True)
  rng = np.random.Generator(np.random.PCG64(args.seed))
  physical_source_indices = np.asarray(
    rng.choice(source_len, size=args.num_sequences, replace=False),
    dtype=np.int64)
  if np.unique(physical_source_indices).size != args.num_sequences:
    raise RuntimeError("Sampling unexpectedly produced duplicate indices")
  sorted_source_indices = np.sort(physical_source_indices)
  physical_positions = np.searchsorted(
    sorted_source_indices, physical_source_indices).astype(np.int64)

  indices_path = output_root / "selected_source_indices_in_output_order.npy"
  sorted_indices_path = output_root / "selected_source_indices_sorted.npy"
  np.save(indices_path, physical_source_indices, allow_pickle=False)
  np.save(sorted_indices_path, sorted_source_indices, allow_pickle=False)

  print("Materializing source-sorted staging dataset", flush=True)
  source_dataset.select(sorted_source_indices).save_to_disk(
    str(sorted_stage), max_shard_size=args.max_shard_size)
  del source_dataset

  print("Writing the final dataset in deterministic shuffled order", flush=True)
  sorted_dataset = datasets.load_from_disk(str(sorted_stage))
  sorted_dataset.select(physical_positions).save_to_disk(
    str(final_stage), max_shard_size=args.max_shard_size)
  del sorted_dataset

  check = datasets.load_from_disk(str(final_stage))
  if len(check) != args.num_sequences:
    raise RuntimeError(f"Final length mismatch: {len(check):,}")
  first_len = len(check[0]["input_ids"])
  last_len = len(check[-1]["input_ids"])
  if first_len != args.sequence_length or last_len != args.sequence_length:
    raise RuntimeError(
      "Packed sequence-length check failed: "
      f"first={first_len}, last={last_len}, expected={args.sequence_length}")
  fingerprint = getattr(check, "_fingerprint", None)
  columns = list(check.column_names)
  del check

  # Rename only after verification.  Readers can never observe a partial cache.
  os.replace(final_stage, final_dataset)
  shutil.rmtree(sorted_stage)

  if validation_link.exists() or validation_link.is_symlink():
    if (not validation_link.is_symlink()
        or validation_link.resolve() != validation_source):
      raise FileExistsError(
        f"Validation cache path already has another target: {validation_link}")
  else:
    validation_link.symlink_to(validation_source, target_is_directory=True)

  manifest = {
    "format_version": 1,
    "sampling": "uniform_without_replacement",
    "rng": "numpy.random.PCG64",
    "numpy_version": np.__version__,
    "datasets_version": datasets.__version__,
    "seed": args.seed,
    "source_dataset": str(source),
    "source_state_sha256": _sha256_file(source / "state.json"),
    "source_num_sequences": source_len,
    "output_dataset": str(final_dataset),
    "output_fingerprint": fingerprint,
    "output_columns": columns,
    "output_num_sequences": args.num_sequences,
    "sequence_length": args.sequence_length,
    "output_num_tokens": args.num_sequences * args.sequence_length,
    "selected_indices_file": indices_path.name,
    "selected_indices_sha256": _sha256_file(indices_path),
    "sorted_indices_file": sorted_indices_path.name,
    "sorted_indices_sha256": _sha256_file(sorted_indices_path),
    "validation_dataset": str(validation_source),
    "validation_link": str(validation_link),
    "elapsed_seconds": time.time() - started,
  }
  manifest_stage = output_root / ".subset_manifest.json.tmp"
  manifest_path = output_root / "subset_manifest.json"
  manifest_stage.write_text(json.dumps(manifest, indent=2) + "\n")
  os.replace(manifest_stage, manifest_path)

  print(json.dumps(manifest, indent=2), flush=True)
  print(f"READY: {final_dataset}", flush=True)


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print(f"SUBSET BUILD FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise
