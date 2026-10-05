"""Sampler implementation for BD3LM block-diffusion models."""

from __future__ import annotations

import numpy as np
import torch

from ..forward_process.utils import sample_categorical
from .base import Sampler


class BD3LMSampler(Sampler):
  """Sampler that mirrors BD3LM's legacy sampling helpers."""

  def __init__(self, config, forward_process=None) -> None:
    self.config = config

  def _nucleus_sample(self, model, p_x0: torch.Tensor) -> torch.Tensor:
    p = getattr(self.config.sampling, 'p_nucleus', 1.0)
    if p == 1.0:
      return p_x0
    block_size = model.block_size
    p_x0_block = p_x0[:, -block_size:].clone()
    sorted_probs, sorted_indices = p_x0_block.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_block.scatter_(-1, sorted_indices, sorted_probs)
    p_x0_block /= p_x0_block.sum(-1, keepdim=True)
    p_x0[:, -block_size:] = p_x0_block
    return p_x0

  def compute_posterior(self, model, x, t, dt, p_x0=None):
    _, move_chance_t = model.noise.forward(t)
    _, move_chance_s = model.noise.forward(t - dt)
    sigma_t = model._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if getattr(self.config.sampling, 'kv_cache', False):
        p_x0 = model.forward(
          x[:, -model.block_size:], sigma_t,
          sample_mode=True).to(torch.float64)
      else:
        p_x0 = model.forward(x, sigma_t, sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -model.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(model, p_x0)

    if getattr(self.config.sampling, 'first_hitting', True):
      x_block = sample_categorical(p_x0)
      mask_region = (x[:, -model.block_size:] == model.mask_id)
      num_masked = mask_region.sum(-1)
      replace_idx = []
      for row_mask in mask_region:
        positions = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
          replace_idx.append(torch.tensor(0, device=x.device))
        else:
          choice = torch.randint(
            0, positions.numel(), (1,), device=x.device).squeeze()
          replace_idx.append(positions[choice])
      replace_idx = torch.stack(replace_idx)
      mask = (torch.arange(model.block_size, device=x.device)
              == replace_idx[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -model.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, model.mask_id] = mask_prob.squeeze(-1)
      x_block = sample_categorical(q_xs)
    copy_flag = (x[:, -model.block_size:] != model.mask_id).to(x.dtype)
    x_block = copy_flag * x[:, -model.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-model.block_size], x_block), dim=-1)

    if (getattr(self.config.sampling, 'kv_cache', False)
        and model.mask_id not in x_block):
      _ = model.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    return p_x0, x_new


  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy

  def _check_stop_conds(self, model, x):
    stop = False
    truncate_idx = None
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True
    if getattr(self.config.sampling, 'var_length', False):
      eos_positions = torch.where(x == model.tokenizer.eos_token_id)
      if len(eos_positions[0]) > 1:
        stop = True
        truncate_idx = min(eos_positions[1][1] + 1, x.shape[1])
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)
    return stop, x

  def _semi_ar_sampler(self, model, num_steps, seqlen, inject_bos):
    n_samples = 1
    ones = torch.ones((n_samples, 1), dtype=model.dtype, device=model.device)
    if getattr(self.config.sampling, 'kv_cache', False):
      reset_fn = getattr(model.backbone, 'reset_kv_cache', None)
      if callable(reset_fn):
        reset_fn()
    sampling_steps = 0
    mdlm_semi_ar = (
      getattr(self.config.algo, 'name', '') == 'mdlm'
      and self.config.model.length > model.block_size)
    num_strides = seqlen // model.block_size
    x_accum = None

    for stride_num in range(num_strides):
      if stride_num == 0:
        x_accum = model.prior_sample(n_samples, model.block_size)
        if inject_bos:
          x_accum[:, 0] = model.tokenizer.bos_token_id
      else:
        stride = 512 if (mdlm_semi_ar and stride_num > 0) else model.block_size
        x = model.prior_sample(n_samples, stride)
        x_accum = torch.cat((x_accum, x), dim=1)

      end_idx = (stride_num + 1) * model.block_size
      start_idx = max(end_idx - 1024, 0)
      fwd_idx = torch.arange(start_idx, end_idx, device=model.device)
      if mdlm_semi_ar and stride_num > 0:
        fwd_idx = torch.arange(
          512 * stride_num,
          512 * stride_num + model.block_size,
          device=model.device)

      dt = 1 / max(num_steps, 1)
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=model.device)
      current_t = torch.tensor(1.0, device=model.device, dtype=torch.float32)

      for idx in range(num_steps):
        if model.mask_id not in x_accum:
          break
        if getattr(self.config.sampling, 'first_hitting', True):
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == model.mask_id).sum(-1).item()
          num_masked = max(num_masked, 1)
          current_t *= u ** (1 / num_masked)
        else:
          current_t = timesteps[idx]
        p_x0_cache, x_next = self.compute_posterior(
          model=model,
          x=x_accum[:, fwd_idx],
          t=current_t * ones,
          dt=dt,
          p_x0=p_x0_cache)
        if p_x0_cache is None:
          sampling_steps += 1
        x_accum[:, fwd_idx] = x_next

      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(model, x_accum)
        if stop:
          return None, None
    return x_accum, sampling_steps


  def _sample_once(self, model, num_steps, eps, inject_bos):
    seqlen = self.config.model.length
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
      sample, nfes = self._semi_ar_sampler(
        model=model,
        num_steps=num_steps,
        seqlen=seqlen,
        inject_bos=inject_bos)
      if sample is not None:
        return sample, nfes
      attempts += 1
    raise ValueError('Sampling failed.')

  def _update_metrics(self, model, nfes):
    if hasattr(model.metrics, 'nfes'):
      model.metrics.nfes.update(nfes)
    if hasattr(model.metrics, 'gen_nfes'):
      model.metrics.gen_nfes.append(nfes)

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    if num_samples is None:
      num_samples = self.config.loader.eval_batch_size
    if num_steps is None:
      num_steps = getattr(self.config.algo, 'T', 0)
    if inject_bos is None:
      inject_bos = getattr(self.config.sampling, 'inject_bos', True)

    samples = []
    nfes_records = []
    for _ in range(num_samples):
      sample, nfes = self._sample_once(
        model=model,
        num_steps=num_steps,
        eps=eps,
        inject_bos=inject_bos)
      samples.append(sample)
      if nfes is not None:
        nfes_records.append(nfes)
        self._update_metrics(model, nfes)
    return torch.cat(samples, dim=0)


__all__ = ["BD3LMSampler"]
