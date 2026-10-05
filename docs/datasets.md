# Datasets, downloads, and caching

SwissAI-DLM can generate synthetic tokens, download datasets through Hugging
Face, stream large corpora, or train from pretokenized files stored locally.
This guide describes each path and the recommended layout on CSCS Clariden.

## Choose a data path

| Goal | Recommended path | Network needed during training? |
| --- | --- | --- |
| Verify an installation | `data=synthetic` or `data=synthetic-gidd` | No |
| Use a small or medium public dataset | A config such as `data=wikitext2` or `data=tinystories` | First run only |
| Avoid materializing a very large corpus | A streaming config such as `data=fineweb-edu` | Yes |
| Run the SwissAI GIDD experiments | Download and pre-cache `data=nemotron-cc-pretok` | No, after preparation |
| Reproduce a fixed token budget | Build a deterministic packed subset | No, after preparation |

Dataset configurations live in `configs/data/`. Print the resolved
configuration before an expensive run:

```bash
python -m discrete_diffusion \
  data=nemotron-cc-pretok model=gidd_hf algo=gidd \
  --cfg job --resolve
```

## Recommended Clariden storage layout

Keep downloads, processed Arrow datasets, and Hugging Face metadata out of
the home directory:

```bash
export DATA_ROOT="${SCRATCH}/SwissAI-DLM-data"
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DATA_ROOT}/cache/discrete_diffusion"
export HF_HOME="${DATA_ROOT}/cache/hf"
export NEMOTRON_PRETOK_DIR="${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok"

mkdir -p \
  "${DISCRETE_DIFFUSION_SCRATCH_DIR}" \
  "${HF_HOME}" \
  "${NEMOTRON_PRETOK_DIR}" \
  "${DATA_ROOT}/outputs" \
  logs
```

`DISCRETE_DIFFUSION_SCRATCH_DIR` controls the processed dataset directories
created by this project. `HF_HOME` controls Hugging Face downloads and
metadata. `NEMOTRON_PRETOK_DIR` identifies the downloaded Nemotron Parquet
files. These are separate layers and can require substantial space.

Scratch storage is subject to CSCS retention policies. Copy irreplaceable
manifests, final checkpoints, and experiment metadata to durable project
storage.

## Option 1: synthetic data

Synthetic data is the fastest installation check because it requires no
credentials or download:

```bash
python -m discrete_diffusion \
  data=synthetic \
  model=tiny \
  algo=mdlm \
  strategy=single-device \
  trainer.devices=1 \
  trainer.max_steps=2 \
  trainer.num_sanity_val_steps=0 \
  loader.global_batch_size=8 \
  loader.eval_global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=0 \
  eval.generate_samples=false \
  checkpointing.save_dir=. \
  hydra.run.dir="${DATA_ROOT:-.}/outputs/synthetic-smoke" \
  '~callbacks' \
  '~wandb'
```

Use `data=synthetic-gidd` with a compatible GIDD model and algorithm when
testing the GIDD training path. Synthetic data checks code and hardware; it
does not measure model quality.

## Option 2: automatic Hugging Face download and cache

Most configs call `datasets.load_dataset` automatically. For example:

```bash
python -m discrete_diffusion \
  data=wikitext2 \
  model=tiny \
  algo=mdlm \
  trainer.max_steps=2 \
  hydra.run.dir="${DATA_ROOT}/outputs/wikitext2-smoke" \
  '~wandb'
```

On the first run, Hugging Face downloads the raw dataset and the project
tokenizes/packs it. The processed result is saved below the selected data
config's `cache_dir`. Later runs reuse it when all cache-defining settings
match.

The processed cache name includes important settings such as:

- dataset and split;
- model sequence length (`model.length`);
- wrapped versus unwrapped packing;
- EOS and special-token insertion; and
- minimum-length or chunking options.

Changing one of those settings can create another large cache. Inspect the
resolved config and available space before launching.

### Authentication for gated datasets

Authenticate without putting a token in Git:

```bash
export HF_TOKEN='<read-token>'
```

Alternatively, use the Hugging Face CLI inside the container:

```bash
huggingface-cli login
```

Do not write a real token into a tracked YAML, shell script, notebook, or EDF
file. For a batch job, export `HF_TOKEN` before `sbatch`, or place it in a
private token file readable only by you.

## Option 3: streaming

Streaming avoids downloading and materializing an entire raw corpus. Existing
streaming choices include:

```text
data=openwebtext-streaming
data=lm1b-streaming
data=fineweb-edu
```

Select one exactly like any other Hydra data config:

```bash
python -m discrete_diffusion \
  data=fineweb-edu \
  model=small \
  algo=mdlm \
  trainer.max_steps=1000
```

Streaming is useful when disk is limited, but it has tradeoffs:

- compute nodes need reliable network access;
- repeated runs may fetch data again;
- exact map-style sampling and some resume/tracing features are unavailable;
- reproducibility depends on the remote dataset revision and stream order.

For a long production run, prefer a pinned dataset revision and a local,
materialized cache when practical.

## Option 4: pretokenized Nemotron on Clariden

The SwissAI GIDD workflow uses
`dvruette/gidd-nemotron-cc-pretok`. The recommended path is to download its
Parquet files once, then build the packed Arrow cache once on a large CPU
node.

### 1. Enter the project container

Follow the [Clariden container guide](clariden-container.md), activate
`venv-docker`, and verify that `HF_TOKEN` is available.

### 2. Download the Parquet files

From the repository root:

```bash
mkdir -p logs
export HF_HOME="${SCRATCH}/SwissAI-DLM-data/cache/hf"
export NEMOTRON_PRETOK_DIR="${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok"
export HF_TOKEN='<read-token>'
sbatch scripts/data/download_dataset.sh
```

The job downloads to:

```text
${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok
```

It verifies that all 4,608 expected Parquet files exist. Check progress with:

```bash
squeue -u "${USER}"
tail -f gidd_download_<job-id>.log
```

After completion:

```bash
export NEMOTRON_PRETOK_DIR="${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok"
find "${NEMOTRON_PRETOK_DIR}" -type f -name '*.parquet' | wc -l
du -sh "${NEMOTRON_PRETOK_DIR}"
```

### 3. Build the reusable packed cache

Packing is CPU- and memory-intensive. Submit the provided CPU job instead of
having every GPU run rebuild the data:

```bash
mkdir -p logs
sbatch scripts/data/precache_nemotron_data.run
```

The launcher requests a large CPU node and writes the packed cache below:

```text
${SCRATCH}/SwissAI-DLM-data/cache/discrete_diffusion/nemotron-cc-pretok
```

Its defaults target 2,048-token GIDD training. Cache reuse requires matching
sequence length, wrapping, EOS, and special-token settings. Override worker
counts when a different CPU or memory allocation is used:

```bash
CACHE_NUM_PROC=64 PACK_NUM_PROC=64 \
  VALID_CACHE_NUM_PROC=1 VALID_PACK_NUM_PROC=1 \
  sbatch scripts/data/precache_nemotron_data.run
```

### 4. Train from the local data

Set both roots in every interactive or batch environment:

```bash
export DISCRETE_DIFFUSION_SCRATCH_DIR="${SCRATCH}/SwissAI-DLM-data/cache/discrete_diffusion"
export NEMOTRON_PRETOK_DIR="${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok"
```

Then select:

```bash
data=nemotron-cc-pretok model=gidd_hf algo=gidd
```

If `NEMOTRON_PRETOK_DIR` is intentionally unavailable, force the loader to
read the Hugging Face dataset directly with:

```bash
data=nemotron-cc-pretok data.pretok_local_dir=null
```

That fallback requires network access and authentication and is less suitable
for repeated multi-node training.

## Option 5: deterministic packed subset

After building the full 2,048-token Nemotron cache, create the reproducible
approximately two-billion-token subset used by the supplied experiments:

```bash
mkdir -p logs
sbatch scripts/data/build_nemotron_2b_reproducible_subset.sbatch
```

The builder samples packed rows without replacement using NumPy PCG64 and
seed 4. It saves the selected indices and a JSON manifest containing hashes,
dataset sizes, versions, and output metadata. It refuses to replace an
existing output directory.

Run the small builder validation first when changing this pipeline:

```bash
mkdir -p logs
sbatch scripts/data/smoke_build_reproducible_subset.sbatch
```

## Offline reuse

After the raw/tokenizer files and processed cache are present, an offline job
can prevent accidental network access:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Test offline mode with a short job before committing a large allocation. A
missing tokenizer, dataset revision, or processed cache will fail instead of
being downloaded.

## Inspecting and maintaining caches

Useful read-only checks:

```bash
du -sh "${HF_HOME}" "${DISCRETE_DIFFUSION_SCRATCH_DIR}"
find "${DISCRETE_DIFFUSION_SCRATCH_DIR}" -name state.json -print
find "${NEMOTRON_PRETOK_DIR}" -type f -name '*.parquet' | wc -l
```

Do not delete a broad cache root while jobs are running. Before removing a
specific processed dataset, confirm that no active or resumable experiment
depends on its exact path and configuration.

## Adding another Hugging Face dataset

For a standard text dataset supported by `datasets.load_dataset`, copy the
closest YAML file under `configs/data/`, change `train`, `valid`, tokenizer,
cache directory, streaming mode, and packing settings, then select the new
filename as `data=<name>`.

The generic loader expects conventional train/validation/test splits and a
text-compatible schema. Pretokenized data, unusual split names, multiple text
columns, or custom preprocessing may require a dedicated branch in
`src/discrete_diffusion/data/loaders.py`. Test new data with a tiny model and
two training steps before launching a full run.

## Troubleshooting

- **Home quota fills:** verify `HF_HOME` and
  `DISCRETE_DIFFUSION_SCRATCH_DIR` before importing or training.
- **401/403 from Hugging Face:** accept the dataset terms if required and
  verify that `HF_TOKEN` is a readable token with dataset access.
- **No Parquet files found:** verify `NEMOTRON_PRETOK_DIR` inside the
  container and ensure the scratch path is mounted by EDF.
- **Cache is rebuilt unexpectedly:** compare the resolved data config,
  especially model length, wrapping, EOS/special-token flags, and chunking.
- **Packing runs out of memory:** reduce `CACHE_NUM_PROC`, `PACK_NUM_PROC`, or
  `PACK_WRITER_BATCH_SIZE` and resubmit the CPU job.
- **Offline mode fails:** temporarily unset the offline variables, fetch the
  missing dataset/tokenizer artifact, then retry offline.
