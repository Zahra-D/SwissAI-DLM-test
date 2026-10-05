# Script index

Scripts are grouped by intent so production launchers are not mixed with
one-off diagnostics. Run them from the repository root unless a script says
otherwise. Most commands expect the Clariden `uni-d2` EDF environment and a
`venv-docker` created with the [container guide](../docs/clariden-container.md).

No launcher contains an authentication token. Export credentials such as
`HF_TOKEN` and `WANDB_API_KEY` in your shell or use the service CLI. Before
submitting, review the SLURM account, partition, node/GPU counts, paths, W&B
destination, token budget, and resume behavior.

## `training/`

- `submit_scion_single_run.sh`: short single-run GIDD-HF + SCION smoke job.
- `submit_scion_hparam_sweep.sh`: main optimizer/performance sweep.
- `submit_scion_hparam_sweep_newImg*.sh`: newer Section B.3 parameterization
  sweeps retained for experiment reproduction.
- `submit_scion_b3_*.sh`: base-model Section B.3 grids and subset wrapper.
- `submit_scion_budget_*.sh` and
  `submit_scion_extending_budget_lr_sweep.sh`: token-budget/LR experiments.
- `submit_scion_modelsize_lr_transfer.sh`: model-size LR transfer study.
- `submit_scion_rho_sigma_*.sh`: batch-scaling trace measurements.
- `submit_scion_global_time_ab.sh`: local-vs-global time-sampling comparison.
- `submit_scion_trace_modelsize_sweep.sh`: trace collection by model size.
- `submit_fsdp_8b_throughput_smoke.sh`: short 8B AdamW/FSDP benchmark.
- `submit_pipeline_1f1b_8b_debug.sh`: experimental 1F1B pipeline fit test.
- `validate_*.sh` / `validate_checkpoints_single_gpu.py`: checkpoint
  validation tools.
- `run_scion_*`: SCION trace/fitting helpers.
- `00_train.run` and `CORRECT_FORMAT_FOR_JOB_SUBMISSION.run`: historical
  templates; prefer a maintained launcher above for new experiments.
- `legacy/`: superseded personal/debug sweep variants kept only for exact
  experiment provenance. Do not use them as templates.

Sweep launchers that support `DRY_RUN` default to preview mode. Always run the
preview first and inspect every generated `sbatch` command.

## `data/`

- `download_dataset.sh`: download the gated pretokenized Nemotron dataset;
  requires `HF_TOKEN` or the configured token file.
- `precache_nemotron_data.py` / `.run`: prepare and pack the cache once on a
  large-memory CPU node.
- `build_reproducible_packed_subset.py` and its `.sbatch` wrapper: build a
  deterministic packed subset.
- `smoke_build_reproducible_subset.sbatch`: small subset-builder check.
- `inspect_data_cursor.py` / `.sbatch`: inspect fault-tolerant data state in a
  checkpoint; set `CHECKPOINT_PATH` for the batch job.

Set `NEMOTRON_PRETOK_DIR` to the downloaded dataset directory when selecting
`data=nemotron-cc-pretok`.

## `analysis/`

This directory contains W&B export/copy/report tools, SCION constant and
batch-noise fitting, and LR/validation-NLL plotting. W&B operations default to
read-only except scripts whose names explicitly contain `copy`. Pass entity,
project, group, and output paths explicitly when possible.

## `debug/`

Short FSDP, data-loader, trace-loader, and resumable-sampler diagnostics. They
are correctness tests or memory/startup probes, not production recipes.

## `maintenance/`

Cleanup scripts can modify local output trees or delete W&B runs. Read them in
full before use. W&B deletion wrappers require `DELETE_CONFIRMED=1` and verify
expected run metadata, but the deletion is still irreversible.

## Adding a launcher

1. Put it in the matching directory and give it a descriptive name.
2. Resolve the repository from `BASH_SOURCE[0]`; do not hard-code a username.
3. Read cluster-specific paths and credentials from environment variables.
4. Use `set -euo pipefail` and quote path variables.
5. Default sweeps or destructive actions to preview/refusal mode.
6. Add a short entry to this index and run `bash -n` before committing.
