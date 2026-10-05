#!/usr/bin/env python3
"""Standalone, single-GPU checkpoint validation.

Does NOT import or modify `discrete_diffusion.train`. It reconstructs the
tokenizer / dataloaders / model / trainer the same way `train.py` does (same
calls, same order) so that `val/nll`, `val/ppl`, `val/bpd` are computed by the
exact same code path used during training -- just forced onto a single GPU
and with `enable_checkpointing=False` / `callbacks=[]` / `logger=False` so
this can never write anything into a run's original output directory.

For each run directory under --outputs_dir:
  - loads that run's own recorded `.hydra/config.yaml` (exact dataset,
    batch size, model, algo settings used during training)
  - overrides only what's required to validate on 1 GPU:
      trainer.devices=1, trainer.num_nodes=1, strategy=SingleDeviceStrategy
    (loader.global_batch_size / loader.batch_size / loader.eval_batch_size
    are left untouched -- accumulate_grad_batches is a `${div_up:...}`
    interpolation in these configs, so it recomputes itself against the new
    devices/num_nodes automatically and the batch-size assertion in
    get_dataloaders() still holds)
  - loads <run>/dummy_checkpoints/checkpoints/<ckpt_name> (last.ckpt by
    default) via trainer.validate(..., ckpt_path=...)
  - appends one row to the output CSV

IMPORTANT CAVEAT (read before comparing numbers to what was logged live
during training): validation loss here is a Monte Carlo estimate over a
randomly sampled diffusion timestep `t` per example (see
`Diffusion._sample_t` / `noise_schedules/gidd_easydel.py:sample_t`, both call
bare `torch.rand(...)` off the live global RNG state). Training never
seeds this deterministically before a validation pass, so the exact
`val/nll` figure logged mid-training is not reproducible bit-for-bit by any
after-the-fact script, regardless of GPU count -- it depended on wherever
the RNG happened to be after N training steps. This script instead calls
`L.seed_everything(config.seed)` once per checkpoint right before
validating, which makes ITS OWN numbers reproducible across re-runs and
directly comparable across the checkpoints/runs it evaluates (removing
random-t noise as a confound between them), at the cost of not bit-matching
the historical training-log value.
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import torch

print(f"[env check] PYTORCH_CUDA_ALLOC_CONF="
      f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')!r} "
      f"(as seen inside this process)", flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

import discrete_diffusion.__main__  # noqa: E402  (side effect: registers OmegaConf resolvers)
from discrete_diffusion.data import get_dataloaders, get_tokenizer  # noqa: E402
from discrete_diffusion import utils  # noqa: E402


def _disable_torch_compile(fn):
  compiler = getattr(torch, 'compiler', None)
  disable = getattr(compiler, 'disable', None) if compiler is not None else None
  if disable is None:
    disable = torch._dynamo.disable
  return disable(fn)


class _MemoryProbeCallback(L.Callback):
  """Diagnostic only: logs CUDA memory every N validation batches so we can
  see *where* it starts climbing, without guessing from code reading alone."""

  def __init__(self, every_n_batches: int = 20):
    self.every_n_batches = every_n_batches

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                               batch_idx, dataloader_idx=0):
    if (batch_idx + 1) % self.every_n_batches == 0:
      allocated = torch.cuda.memory_allocated() / 1e9
      reserved = torch.cuda.memory_reserved() / 1e9
      print(f"    [mem probe] batch {batch_idx + 1}: "
            f"allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB",
            flush=True)


def _find_run_dirs(outputs_dir: Path):
  run_dirs = []
  for child in sorted(outputs_dir.iterdir()):
    if not child.is_dir():
      continue
    if (child / ".hydra" / "config.yaml").exists():
      run_dirs.append(child)
  return run_dirs


def validate_one_run(run_dir: Path, ckpt_name: str, scratch_dir: Path,
                      generate_samples: bool, debug_memory: bool = False,
                      memory_log_every: int = 20, max_val_batches=None):
  cfg = omegaconf.OmegaConf.load(run_dir / ".hydra" / "config.yaml")

  ckpt_path = run_dir / "dummy_checkpoints" / "checkpoints" / ckpt_name
  if not ckpt_path.exists():
    return {"run_name": run_dir.name, "status": f"missing checkpoint: {ckpt_path}"}

  # --- single-GPU overrides only ---
  cfg.trainer.devices = 1
  cfg.trainer.num_nodes = 1
  cfg.strategy = omegaconf.OmegaConf.create({
      "_target_": "lightning.pytorch.strategies.SingleDeviceStrategy",
      "device": "cuda:0",
  })
  cfg.eval.generate_samples = generate_samples
  if max_val_batches is not None:
    # Diagnostic only: caps how many validation batches run, so we can
    # reproduce the memory-growth pattern cheaply instead of burning a full
    # 2149-batch pass every time.
    cfg.trainer.limit_val_batches = max_val_batches

  run_scratch = scratch_dir / run_dir.name
  run_scratch.mkdir(parents=True, exist_ok=True)

  torch.set_float32_matmul_precision("high")
  L.seed_everything(cfg.seed)

  tokenizer = get_tokenizer(cfg)
  algo_cls = hydra.utils.get_class(cfg.algo._target_)

  # NOTE: get_dataloaders(..., skip_train=True) is currently broken in
  # data/loaders.py -- `mp_context` is only assigned inside the `else`
  # branch of `if skip_train:` but is also read while building the (separate)
  # valid_loader, so skip_train=True raises UnboundLocalError. Working around
  # it here by not skipping train construction, rather than patching that
  # shared file. This costs a bit of extra time building an unused train
  # dataset/loader (never iterated) but has zero effect on validation output.
  _, valid_ds = get_dataloaders(cfg, tokenizer, skip_train=False)

  if cfg.training.finetune_path != "":
    assert utils.fsspec_exists(cfg.training.finetune_path)
    model = algo_cls.load_from_checkpoint(
        cfg.training.finetune_path, tokenizer=tokenizer, config=cfg)
  else:
    model = algo_cls(cfg, tokenizer=tokenizer)

  if omegaconf.OmegaConf.select(cfg, "training.torch_compile", default=False):
    model.log = _disable_torch_compile(model.log)
    model.log_dict = _disable_torch_compile(model.log_dict)
    model = torch.compile(model)

  callbacks = [_MemoryProbeCallback(memory_log_every)] if debug_memory else []
  trainer = L.Trainer(
      **cfg.trainer,
      default_root_dir=str(run_scratch),
      callbacks=callbacks,
      logger=False,
      enable_checkpointing=False,
      strategy=hydra.utils.instantiate(cfg.strategy),
  )

  if debug_memory:
    torch.cuda.memory._record_memory_history(max_entries=200_000)

  try:
    results = trainer.validate(model, dataloaders=valid_ds, ckpt_path=str(ckpt_path))
  except torch.cuda.OutOfMemoryError:
    if debug_memory:
      snap_path = run_scratch / "oom_snapshot.pickle"
      torch.cuda.memory._dump_snapshot(str(snap_path))
      print(f"    [mem probe] dumped OOM snapshot to {snap_path}", flush=True)
    raise
  finally:
    if debug_memory:
      snap_path = run_scratch / "final_snapshot.pickle"
      torch.cuda.memory._dump_snapshot(str(snap_path))
      torch.cuda.memory._record_memory_history(enabled=None)
      print(f"    [mem probe] dumped snapshot to {snap_path}", flush=True)

  metrics = dict(results[0]) if results else {}
  metrics["run_name"] = run_dir.name
  metrics["ckpt_path"] = str(ckpt_path)
  metrics["global_step"] = trainer.global_step
  metrics["status"] = "ok"

  del model, trainer, valid_ds
  gc.collect()
  torch.cuda.empty_cache()

  return metrics


def main():
  p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--outputs_dir", type=Path, required=True,
                  help="Directory containing one subdir per run "
                       "(e.g. outputs/scion_new_runs_2026-07-27).")
  p.add_argument("--ckpt_name", default="last.ckpt")
  p.add_argument("--output_csv", type=Path, default=None,
                  help="Defaults to <outputs_dir>/validation_results.csv")
  p.add_argument("--scratch_dir", type=Path, default=None,
                  help="Scratch dir for Lightning's default_root_dir per run "
                       "(logs only -- no checkpoints are ever written here). "
                       "Defaults to <outputs_dir>/validation_scratch")
  p.add_argument("--generate_samples", action="store_true",
                  help="Also run config.eval sample generation (slower; "
                       "irrelevant to val/nll and val/ppl). Off by default.")
  p.add_argument("--only_run", default=None,
                  help="Substring filter -- only validate run dirs whose "
                       "name contains this (for fast, targeted debugging).")
  p.add_argument("--debug_memory", action="store_true",
                  help="Log CUDA memory every --memory_log_every batches and "
                       "dump a torch.cuda.memory snapshot (for OOM root-causing "
                       "with torch's memory viz tooling). Off by default.")
  p.add_argument("--memory_log_every", type=int, default=20)
  p.add_argument("--max_val_batches", type=int, default=None,
                  help="Diagnostic only: caps trainer.limit_val_batches to "
                       "this many batches so a repro run is cheap. Leave "
                       "unset for real (full-dataset) validation runs.")
  args = p.parse_args()

  output_csv = args.output_csv or (args.outputs_dir / "validation_results.csv")
  scratch_dir = args.scratch_dir or (args.outputs_dir / "validation_scratch")
  scratch_dir.mkdir(parents=True, exist_ok=True)

  run_dirs = _find_run_dirs(args.outputs_dir)
  if args.only_run is not None:
    run_dirs = [d for d in run_dirs if args.only_run in d.name]
  print(f"Found {len(run_dirs)} run dirs under {args.outputs_dir}")

  rows = []
  fieldnames = set(["run_name", "ckpt_path", "global_step", "status"])

  for i, run_dir in enumerate(run_dirs):
    print(f"[{i + 1}/{len(run_dirs)}] validating {run_dir.name} ...", flush=True)
    try:
      row = validate_one_run(
          run_dir, args.ckpt_name, scratch_dir, args.generate_samples,
          debug_memory=args.debug_memory,
          memory_log_every=args.memory_log_every,
          max_val_batches=args.max_val_batches)
    except Exception as e:  # noqa: BLE001
      row = {"run_name": run_dir.name, "status": f"error: {e}"}
      print(f"  FAILED: {e}", flush=True)
      gc.collect()
      torch.cuda.empty_cache()
    rows.append(row)
    fieldnames.update(row.keys())

    # Write incrementally so partial progress survives a crash later in the loop.
    with open(output_csv, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
      writer.writeheader()
      for r in rows:
        writer.writerow(r)
    print(f"  status={row.get('status')} -> {output_csv}", flush=True)

  print(f"\nDone. Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
  main()
