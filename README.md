# SwissAI-DLM / UNI-D²

SwissAI-DLM is a research codebase for training and evaluating discrete
diffusion language models. It extends the upstream
[UNI-D² project](https://github.com/nkalyanv99/UNI-D2) with GIDD models,
SCION optimization, distributed training, reproducible data preparation, and
Clariden/SLURM experiment workflows.

## What is included

- Hydra + Lightning training through `python -m discrete_diffusion`
- MDLM, UDLM, BD3LM, FlexMDM, GIDD, SEDD, PartitionMDLM, CANDI, and AR methods
- DiT, Hugging Face GPT-2, and GIDD Hugging Face model configurations
- AdamW and SCION optimizers
- DDP, FSDP, and experimental pipeline-parallel strategies
- Reproducible data preparation and Clariden job launchers

## Installation

### Requirements

- Linux
- Python 3.11 recommended (Python 3.9 or newer is declared by the package)
- An NVIDIA CUDA GPU for training
- Git
- Enough local or scratch storage for datasets, checkpoints, and caches

Large training runs require a multi-GPU machine or a SLURM cluster. The
smoke test below uses generated data and one GPU.

### CSCS Clariden: use the shared project image

This is the recommended setup for SwissAI project members. Clone the
repository in your CSCS home directory:

```bash
cd "${HOME}"
git clone <repository-url> SwissAI-DLM
cd SwissAI-DLM
```

Copy the shared 17 GB image to your scratch space. The source is readable by
members of CSCS project `ab035`:

```bash
mkdir -p "${SCRATCH}/ce-images"
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M \
  "${SCRATCH}/ce-images"
cp /capstor/store/cscs/swissai/ab035/ars/DLM-docker/uni-d2-updated.sqsh \
  "${SCRATCH}/ce-images/"
```

Create the EDF environment from the repository template:

```bash
mkdir -p "${HOME}/.edf"
cp containers/uni-d2.toml.example "${HOME}/.edf/uni-d2.toml"
sed -i "s/<username>/${USER}/g" "${HOME}/.edf/uni-d2.toml"
```

Start an interactive allocation using the image:

```bash
srun --environment=uni-d2 -A ab035 --pty bash
```

Inside the allocation, layer this checkout on top of the packages already in
the image:

```bash
cd "${HOME}/SwissAI-DLM"
python -m venv --system-site-packages venv-docker
source venv-docker/bin/activate
python -m pip install -e . --no-deps
```

Verify the environment before training:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
python -c 'import flash_attn, liger_kernel; print("CUDA extensions available")'
```

Then run the synthetic command in [Quick start](#quick-start). The
[full Clariden guide](docs/clariden-container.md) also covers scratch
directories, image rebuilding, submitted jobs, and common failures.

### Standard Python environment

For another CUDA machine or cluster, clone the repository and create a normal
virtual environment:

```bash
git clone <repository-url> SwissAI-DLM
cd SwissAI-DLM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Check that PyTorch can see the GPU:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
```

`CUDA available` must be `True` before running the smoke test. If it is
false, verify that the NVIDIA driver and installed PyTorch build are
compatible.

Some model configurations use optional CUDA extensions. Install these only
when needed and when they are supported by the machine:

```bash
python -m pip install -e '.[flash-attn,liger]'
```

## Quick start

First run a two-step smoke test. It uses generated tokens and a tiny model,
requires no dataset download, and disables W&B, sampling, and checkpoint
callbacks:

```bash
python -m discrete_diffusion \
  data=synthetic \
  model=tiny \
  algo=mdlm \
  strategy=single-device \
  trainer.devices=1 \
  trainer.max_steps=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=1 \
  trainer.val_check_interval=2 \
  loader.global_batch_size=8 \
  loader.eval_global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=0 \
  eval.generate_samples=false \
  checkpointing.save_dir=. \
  hydra.run.dir=outputs/smoke \
  '~callbacks' \
  '~wandb'
```

A successful run prints the composed configuration, completes two training
steps, and writes local output under `outputs/smoke/`.

To inspect a configuration without training, add Hydra's `--cfg job` flag:

```bash
python -m discrete_diffusion --cfg job data=synthetic model=tiny algo=mdlm
```

## Run a real experiment

Hydra composes the selected files under `configs/`. Values supplied on the
command line override their defaults. For example, the following selects
OpenWebText, MDLM, a small DiT model, and four GPUs:

```bash
export DISCRETE_DIFFUSION_SCRATCH_DIR=/path/to/fast-storage/training-data

python -m discrete_diffusion \
  data=openwebtext-split \
  model=small \
  algo=mdlm \
  strategy=ddp \
  trainer.devices=4 \
  loader.batch_size=32 \
  hydra.run.dir=/path/to/fast-storage/outputs/mdlm-openwebtext
```

This is a real multi-GPU job and may download data. Adapt its batch size,
output path, device count, and duration before running it.

W&B logging is enabled by the default configuration. Run `wandb login`
before a real experiment, or append the quoted `'~wandb'` override to
disable W&B. Never put an API token in a committed config or script.

Read [the configuration reference](docs/configuration.md) before starting a
large run. It describes the config groups, available choices, important
settings, and batch-size calculation.

For dataset selection, Hugging Face downloads, scratch-cache setup,
pretokenized Nemotron data, streaming, and offline reuse, see the
[dataset and caching guide](docs/datasets.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | Hydra defaults and selectable data/model/algorithm/runtime configs |
| `src/discrete_diffusion/` | Training, model, optimizer, data, and evaluation code |
| `examples/` | Upstream paper-specific examples |
| `scripts/` | Maintained cluster workflows, grouped by purpose |
| `docs/` | Architecture, configuration, container, and API documentation |
| `notebooks/` | Small analysis and visualization notebooks |

See [the scripts index](scripts/README.md) before using a cluster launcher.
Most launchers default to a dry run where practical, but SLURM account,
storage, and W&B values still need to match your environment.

## Sampling

Given a checkpoint:

```bash
PYTHONPATH=src python -m discrete_diffusion.evaluations.generate_samples \
  checkpoint_path=/path/to/last.ckpt \
  num_samples=16 \
  num_steps=2000
```

## Documentation

Install the development dependencies and build the documentation locally:

```bash
python -m pip install -e '.[dev]'
mkdocs serve
```

## Provenance and license

This repository is derived from
[nkalyanv99/UNI-D2](https://github.com/nkalyanv99/UNI-D2). Please retain the
upstream attribution and cite the original project as described in
[`CITATION.cff`](CITATION.cff). The code is distributed under the
[`MIT License`](LICENSE).
