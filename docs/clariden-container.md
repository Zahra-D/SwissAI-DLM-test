# Clariden container setup

This guide covers the supported CSCS Clariden setup: an Enroot `.sqsh` image
built from NVIDIA PyTorch 25.12 and selected through an EDF environment named
`uni-d2`. The image is a runtime dependency and must not be committed to Git.

## 1. Clone the repository and prepare scratch storage

```bash
git clone <repository-url> SwissAI-DLM
cd SwissAI-DLM

mkdir -p "${SCRATCH}/SwissAI-DLM-data/training-data"
mkdir -p "${SCRATCH}/SwissAI-DLM-data/outputs"
mkdir -p "${SCRATCH}/SwissAI-DLM-data/cache"

ln -s "${SCRATCH}/SwissAI-DLM-data/training-data" training-data
ln -s "${SCRATCH}/SwissAI-DLM-data/outputs" outputs
ln -s "${SCRATCH}/SwissAI-DLM-data/cache" .cache
```

If any link name already exists, inspect it before replacing it. Do not delete
an existing cache or output directory just to make the commands above work.
Scratch storage is subject to CSCS retention policies; move durable artifacts
to project storage.

## 2. Copy the shared project image

The current 17 GB image is stored in project space and is readable by members
of CSCS project `ab035`. Copy it once to your own scratch directory:

```bash
mkdir -p "${SCRATCH}/ce-images"
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M \
  "${SCRATCH}/ce-images"
cp /capstor/store/cscs/swissai/ab035/ars/DLM-docker/uni-d2-updated.sqsh \
  "${SCRATCH}/ce-images/"
ls -lh "${SCRATCH}/ce-images/uni-d2-updated.sqsh"
```

If that project path changes or access is denied, ask a project owner for the
current image location. Do not commit the image to Git.

Copy [`containers/uni-d2.toml.example`](../containers/uni-d2.toml.example) to
`~/.edf/uni-d2.toml` and replace `<username>`:

```bash
mkdir -p "${HOME}/.edf"
cp containers/uni-d2.toml.example "${HOME}/.edf/uni-d2.toml"
sed -i "s/<username>/${USER}/g" "${HOME}/.edf/uni-d2.toml"
```

## 3. Start the environment

From the repository root:

```bash
srun --environment=uni-d2 -A ab035 --pty bash
```

Inside the allocation, create a small virtual environment layered on top of
the container packages:

```bash
python -m venv --system-site-packages venv-docker
source venv-docker/bin/activate
python -m pip install -e . --no-deps
```

The image normally provides Flash Attention and Liger Kernel. Verify them
before installing anything additional:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -c 'import flash_attn, liger_kernel; print("CUDA extensions available")'
```

## 4. Build the image when needed

Only do this if the shared image is unavailable or needs to change. Configure
Podman storage once:

```bash
mkdir -p "${HOME}/.config/containers"
cp containers/storage.conf "${HOME}/.config/containers/storage.conf"
```

Then request a build node and build from the repository root:

```bash
srun -A ab035 --pty bash
podman build -f containers/Dockerfile -t uni-d2:pytorch-25.12 .
enroot import -x mount \
  -o "${SCRATCH}/ce-images/uni-d2-updated.sqsh" \
  podman://uni-d2:pytorch-25.12
```

## 5. Smoke test training

Run this inside the allocation with `venv-docker` activated. It uses one GPU,
generated data, and no W&B account or external dataset:

```bash
export DISCRETE_DIFFUSION_SCRATCH_DIR="${SCRATCH}/SwissAI-DLM-data/training-data"

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
  hydra.run.dir="${SCRATCH}/SwissAI-DLM-data/outputs/smoke" \
  '~callbacks' \
  '~wandb'
```

A successful run completes two optimizer steps and writes output under
`${SCRATCH}/SwissAI-DLM-data/outputs/smoke`.

For submitted jobs, adapt a launcher under `scripts/training/`. Replace the
SLURM account and any environment-specific storage paths before submission.

## Common failures

- **EDF cannot find the image:** verify the expanded path in
  `~/.edf/uni-d2.toml` and that the `.sqsh` file is readable from the compute
  node.
- **A symlink is broken inside the container:** ensure both the home and
  scratch paths are mounted at their original absolute paths.
- **Home quota fills:** confirm `DISCRETE_DIFFUSION_SCRATCH_DIR`, `HF_HOME`,
  outputs, and caches point to scratch storage.
- **CUDA extension import fails:** confirm the image and PyTorch/CUDA versions
  match; rebuilding an extension against a different Torch version can break
  its ABI.
