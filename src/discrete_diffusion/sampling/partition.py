"""Partition sampler for PartitionMDLM with multiple sampling modes."""

from __future__ import annotations

import torch

from ..forward_process.utils import sample_categorical
from .base import Sampler


class PartitionSampler(Sampler):
  """Sampler for PartitionMDLM with naive and efficient sampling modes.
  
  Supports:
  - 'naive': Standard DDPM updates with group_idxs tracking
  - 'efficient-uniform': Uniform token denoising schedule
  - 'efficient-non-uniform': Binomial token denoising schedule
  """

  def __init__(self, config, forward_process=None):
    self.config = config

  def compute_posterior(self, model, x, t, dt, p_x0=None, group_idxs=None,
                        noise_removal_step=False):
    """Compute posterior with group_idxs support for partition tracking."""
    alpha_t = model.noise.alpha_t(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      alpha_s = model.noise.alpha_t(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      log_p_x0 = model.forward(
        x,
        model._sigma_from_alphat(alpha_t),
        group_idxs=group_idxs)
      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      p_x0 = log_p_x0.exp()

    sampled_x0 = sample_categorical(p_x0)
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
    should_denoise_draw = torch.rand_like(x, dtype=torch.float64) < prob_denoise
    is_masked = (x == model.mask_id)
    should_denoise_mask = is_masked & should_denoise_draw
    _x = torch.where(should_denoise_mask, sampled_x0, x)

    if group_idxs is not None:
      out = torch.where(group_idxs == 0, x, _x)
      group_idxs = torch.where(out == x,
                               group_idxs,
                               torch.logical_not(group_idxs))
      return p_x0, out, group_idxs
    else:
      out = torch.where(x != model.mask_id, x, _x)
      return p_x0, out

  def compute_posterior_efficient(self, model, x, t, dt, p_x0,
                                  clean_positions, noisy_positions,
                                  concrete_lengths):
    """Efficient posterior computation for position-based denoising."""
    alpha_t = model.noise.alpha_t(t)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      log_p_x0 = model.forward(
        x,
        model._sigma_from_alphat(alpha_t),
        clean_positions=clean_positions,
        noisy_positions=noisy_positions,
        concrete_lengths=concrete_lengths,
        use_inference_mode=True)
      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      p_x0 = log_p_x0.exp()

    sampled_x0 = sample_categorical(p_x0)
    return sampled_x0

  @torch.no_grad()
  def generate_naive(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Naive generation with group tracking (standard DDPM)."""
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = model.prior_sample(num_samples, model.num_tokens)
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    if not inject_bos:
      raise ValueError("Partition MDLM requires inject_bos=True")
    x[:, 0] = model.tokenizer.bos_token_id
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    predictor = self.config.sampling.predictor

    # Group 0: unmasked, group 1: masked
    group_idxs = torch.ones_like(x, dtype=int)
    group_idxs[:, 0] = 0
    
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
      if predictor == 'ddpm':
        _, x, group_idxs = self.compute_posterior(
          model=model, x=x, t=t, dt=dt, p_x0=None, group_idxs=group_idxs)
      elif predictor == 'ddpm_cache':
        p_x0_cache, x_next, group_idxs = self.compute_posterior(
          model=model, x=x, t=t, dt=dt, p_x0=p_x0_cache, 
          group_idxs=group_idxs)
        if (not torch.allclose(x_next, x) or model.time_conditioning):
          p_x0_cache = None
        x = x_next
      else:
        raise ValueError(f'Unsupported predictor: {predictor}')

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
    _, x, _ = self.compute_posterior(model=model, x=x, t=t0, dt=None,
                                     p_x0=p_x0_cache,
                                     noise_removal_step=True,
                                     group_idxs=group_idxs)
    return x

  @torch.no_grad()
  def generate_efficient_uniform(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Efficient uniform generation (fixed tokens per step)."""
    if num_steps is None:
      num_steps = self.config.sampling.steps
    
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    if not inject_bos:
      raise ValueError("Partition MDLM requires inject_bos=True")

    x = torch.full(size=(num_samples, 1), 
                   fill_value=model.tokenizer.bos_token_id, 
                   device=model.device)
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps

    clean_positions = torch.zeros(size=(num_samples, 1), 
                                  device=model.device, 
                                  dtype=torch.int64)
    noisy_positions = torch.arange(start=1, 
                                   end=self.config.model.length, 
                                   device=model.device, 
                                   dtype=torch.int64)[None
                                    ].repeat(num_samples, 1)
    # Random permutation
    rand = torch.rand_like(noisy_positions, dtype=torch.float32)
    shuffled_indices = rand.argsort(dim=-1)
    noisy_positions = torch.gather(noisy_positions, dim=-1, 
                                   index=shuffled_indices)
    concrete_lengths = torch.ones(size=(num_samples,), 
                                  device=model.device, 
                                  dtype=torch.int64)
    
    if self.config.model.length % num_steps != 0:
      raise ValueError(f"Length {self.config.model.length} must be divisible by steps {num_steps}")
    
    n_tok_per_normal_step = self.config.model.length // num_steps
    all_n_tok_per_step = torch.full(size=(num_steps,), 
                                fill_value=n_tok_per_normal_step)
    # Last step might need more tokens
    all_n_tok_per_step[-1] += (self.config.model.length 
                           - num_steps * n_tok_per_normal_step)
    
    for t, n_tok_per_step in zip(timesteps[:-1], all_n_tok_per_step):
      t = t * torch.ones(x.shape[0], 1, device=model.device)
      noisy_pos_input = noisy_positions[:, :n_tok_per_step]
      denoised_token_values = self.compute_posterior_efficient(
         model=model, x=x, t=t, dt=dt, p_x0=None, 
         clean_positions=clean_positions, 
         noisy_positions=noisy_pos_input, 
         concrete_lengths=concrete_lengths)
      x = torch.cat([x, denoised_token_values], dim=1)
      clean_positions = torch.cat([clean_positions, noisy_pos_input], dim=1)
      noisy_positions = noisy_positions[:, n_tok_per_step:]
      concrete_lengths += n_tok_per_step
    
    # Reorder to original positions
    out = torch.empty_like(x).scatter_(dim=-1, index=clean_positions, src=x)
    return out

  def _gen_eff_non_unif_post_process(self, x, concrete_lengths, 
    n_denoise_per_seq, denoised_token_values, clean_positions, 
    noisy_positions, noisy_pos_input):
    """Post-process for non-uniform efficient generation."""
    new_concrete_lengths = concrete_lengths + n_denoise_per_seq
    n_tok_to_add = new_concrete_lengths.max() - x.shape[1]
    if n_tok_to_add > 0:
      pad = torch.zeros(size=(x.shape[0], n_tok_to_add), 
                        dtype=x.dtype, device=x.device)
      x = torch.cat([x, pad], dim=1)
      clean_positions = torch.cat([clean_positions, pad], dim=1)
    
    for i in range(x.shape[0]):
      if n_denoise_per_seq[i] == 0:
        continue
      x[i, concrete_lengths[i]: new_concrete_lengths[i]] = \
            denoised_token_values[i, :n_denoise_per_seq[i]]
      clean_positions[i, concrete_lengths[i]:new_concrete_lengths[i]] = \
            noisy_pos_input[i, :n_denoise_per_seq[i]]
      noisy_positions[i, :noisy_positions.shape[1] - n_denoise_per_seq[i]] = \
        noisy_positions[i, n_denoise_per_seq[i]:].clone()
    
    return x, clean_positions, new_concrete_lengths

  @torch.no_grad()
  def generate_efficient_non_uniform(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Efficient non-uniform generation (binomial tokens per step)."""
    if num_steps is None:
      num_steps = self.config.sampling.steps
    
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    if not inject_bos:
      raise ValueError("Partition MDLM requires inject_bos=True")

    x = torch.full(size=(num_samples, 1), 
                   fill_value=model.tokenizer.bos_token_id, 
                   device=model.device)
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps

    clean_positions = torch.zeros(size=(num_samples, 1), 
                                  device=model.device, 
                                  dtype=torch.int64)
    noisy_positions = torch.arange(start=1, 
                                   end=self.config.model.length, 
                                   device=model.device, 
                                   dtype=torch.int64)[None
                                    ].repeat(num_samples, 1)
    # Random permutation
    rand = torch.rand_like(noisy_positions, dtype=torch.float32)
    shuffled_indices = rand.argsort(dim=-1)
    noisy_positions = torch.gather(noisy_positions, dim=-1, 
                                   index=shuffled_indices)
    concrete_lengths = torch.ones(size=(num_samples,), 
                                  device=model.device, 
                                  dtype=torch.int64)
    
    alpha_t = model.noise.alpha_t(timesteps[0])
    alpha_s = model.noise.alpha_t(timesteps[0] - dt)
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
    
    for t in timesteps[:-1]:
      t = t * torch.ones(x.shape[0], 1, device=model.device)
      bin_count = torch.ones(size=(num_samples,), 
                             device=prob_denoise.device)
      bin_count *= self.config.model.length
      n_denoise_per_seq = torch.binomial(count=bin_count, 
                                         prob=prob_denoise).to(int)
      n_denoise_per_seq = torch.min(n_denoise_per_seq, 
                self.config.model.length - concrete_lengths)
      denoise_seq_len = torch.max(n_denoise_per_seq).item()
      if denoise_seq_len == 0:
        continue
      
      noisy_pos_input = noisy_positions[:, :denoise_seq_len]
      denoised_token_values = self.compute_posterior_efficient(
         model=model, x=x, t=t, dt=dt, p_x0=None, 
         clean_positions=clean_positions, 
         noisy_positions=noisy_pos_input, 
         concrete_lengths=concrete_lengths)
      
      (x, clean_positions, concrete_lengths) = \
        self._gen_eff_non_unif_post_process(x, concrete_lengths, 
        n_denoise_per_seq, denoised_token_values, clean_positions, 
        noisy_positions, noisy_pos_input)
      
    # Final denoising of remaining masked tokens
    if not torch.all(concrete_lengths == self.config.model.length):
      n_denoise_per_seq = self.config.model.length - concrete_lengths
      noisy_pos_input = noisy_positions[:, :self.config.model.length - concrete_lengths.min()]
      denoised_token_values = self.compute_posterior_efficient(
         model=model, x=x, t=t, dt=dt, p_x0=None, 
         clean_positions=clean_positions, 
         noisy_positions=noisy_pos_input, 
         concrete_lengths=concrete_lengths)
      (x, clean_positions, concrete_lengths) = \
        self._gen_eff_non_unif_post_process(x, concrete_lengths, 
        n_denoise_per_seq, denoised_token_values, clean_positions, 
        noisy_positions, noisy_pos_input)
    
    # Reorder to original positions
    out = torch.empty_like(x).scatter_(dim=-1, index=clean_positions, src=x)
    return out

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Generate samples using configured sampling mode."""
    # Get sampling mode from model config
    sampling_mode = getattr(model, 'sampling_mode', 'naive')
    
    if sampling_mode == 'naive':
      return self.generate_naive(
        model, num_samples=num_samples, num_steps=num_steps, 
        eps=eps, inject_bos=inject_bos)
    elif sampling_mode == 'efficient-uniform':
      return self.generate_efficient_uniform(
        model, num_samples=num_samples, num_steps=num_steps, 
        eps=eps, inject_bos=inject_bos)
    elif sampling_mode == 'efficient-non-uniform':
      return self.generate_efficient_non_uniform(
        model, num_samples=num_samples, num_steps=num_steps, 
        eps=eps, inject_bos=inject_bos)
    else:
      raise ValueError(f'Unknown sampling mode: {sampling_mode}')

