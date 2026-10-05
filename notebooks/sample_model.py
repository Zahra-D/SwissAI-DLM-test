import sys
import textwrap
from pathlib import Path
from typing import Literal

import hydra
import torch
from omegaconf import OmegaConf


def _repo_root() -> Path:
    try:
        here = Path(__file__).resolve()
        return here.parents[1]
    except NameError:
        # Fallback for interactive contexts
        return Path(".").resolve()


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from discrete_diffusion.data import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


def load_model_from_run(run_dir: Path, ckpt_type: Literal["best", "last"] = "best"):
    ckpt_path = (
        Path(run_dir) / "dummy_checkpoints" / "checkpoints" / f"{ckpt_type}.ckpt"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["hyper_parameters"]["config"]
    hydra_cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")

    tokenizer = get_tokenizer(ckpt_cfg)
    algo_target = ckpt_cfg.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    model = algo_cls.load_from_checkpoint(
        ckpt_path,
        config=ckpt_cfg,
        tokenizer=tokenizer,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt_path, ckpt_cfg, hydra_cfg


NUM_SAMPLES = 8
NUM_STEPS = None  # use model default
RUN_DIR = ROOT / "outputs/roneneldan/TinyStories/2026.02.05/155200"
SHOW_STEPS = True
STEP_EVERY = 50
STEP_MAX_SAMPLES = 2
SKIP_SPECIAL_TOKENS = False
EPS = None  # set to a float to override (e.g., 1e-5)
WRAP_WIDTH = 100

def _decode_step(step_idx, num_steps, t_val, z_t, tokenizer, *, max_samples, skip_special_tokens):
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is not None:
        mask_rate = (z_t == mask_id).float().mean().item() * 100
        header = f"Step {step_idx:>4}/{num_steps:<4}  t={t_val:.6f}  mask={mask_rate:>5.1f}%"
    else:
        header = f"Step {step_idx:>4}/{num_steps:<4}  t={t_val:.6f}"
    print("=" * max(60, len(header)))
    print(header)
    print("-" * max(60, len(header)))
    texts = tokenizer.batch_decode(
        z_t.detach().cpu(),
        skip_special_tokens=skip_special_tokens,
    )
    for i, text in enumerate(texts[:max_samples]):
        prefix = f"[{i:02d}] "
        wrapped = textwrap.fill(
            text,
            width=WRAP_WIDTH,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        )
        print(wrapped)
    print()


@torch.no_grad()
def generate_with_steps(model, tokenizer, *, num_samples, num_steps, eps, step_every, max_samples, skip_special_tokens):
    sampler = model._create_sampler()
    if sampler is None or sampler.__class__.__name__ != "GIDDSampler":
        samples = model.generate_samples(num_samples=num_samples, num_steps=num_steps, eps=eps)
        texts = tokenizer.batch_decode(samples.detach().cpu(), skip_special_tokens=True)
        for i, text in enumerate(texts):
            print(f"--- sample {i} ---")
            print(text)
            print()
        return

    if num_steps is None:
        num_steps = model.config.sampling.steps
    if eps is None:
        eps = 1e-5
    inject_bos = getattr(model.config.sampling, "inject_bos", True)

    z_t = model.hybrid_noise.sample_prior((num_samples, model.num_tokens)).to(model.device)
    if inject_bos:
        z_t[:, 0] = model.tokenizer.bos_token_id

    timesteps = torch.linspace(1 - eps, eps, num_steps + 1, device=model.device)
    _decode_step(0, num_steps, timesteps[0].item(), z_t, tokenizer, max_samples=max_samples, skip_special_tokens=skip_special_tokens)

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(num_samples, device=model.device)
        s = timesteps[i + 1] * torch.ones(num_samples, device=model.device)
        z_t = sampler.compute_posterior(model, z_t, t, s)
        if (i + 1) % step_every == 0 or i == num_steps - 1:
            _decode_step(i + 1, num_steps, timesteps[i + 1].item(), z_t, tokenizer, max_samples=max_samples, skip_special_tokens=skip_special_tokens)


model, tokenizer, ckpt_path, ckpt_cfg, hydra_cfg = load_model_from_run(RUN_DIR)
print("Checkpoint:", ckpt_path)
print("Algo:", ckpt_cfg.algo.name)
print("Loss Type:", ckpt_cfg.algo.loss_type)
print("P_u=", ckpt_cfg.algo.p_uniform)

if SHOW_STEPS:
    generate_with_steps(
        model,
        tokenizer,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_STEPS,
        eps=EPS,
        step_every=STEP_EVERY,
        max_samples=STEP_MAX_SAMPLES,
        skip_special_tokens=SKIP_SPECIAL_TOKENS,
    )
else:
    samples = model.generate_samples(num_samples=NUM_SAMPLES, num_steps=NUM_STEPS, eps=EPS)
    texts = tokenizer.batch_decode(samples.detach().cpu(), skip_special_tokens=True)
    for i, text in enumerate(texts):
        print(f"--- sample {i} ---")
        print(text)
        print()
