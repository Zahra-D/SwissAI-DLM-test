"""GIDD sampler for hybrid diffusion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from .base import Sampler


class GIDDSampler(Sampler):
  """Sampler for GIDD (Generalized Iterative Discrete Diffusion) models."""

  def __init__(self, config, forward_process=None):
    self.config = config

  @staticmethod
  def _is_easydel_schedule(model) -> bool:
    return str(getattr(model.config.algo, 'loss_type', '')).lower() == 'gidd_easydel'

  @staticmethod
  def _model_logits(model, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if bool(getattr(model, '_sigma_free_backbone', False)):
      return model.backbone(z_t)
    alpha_t = model._loglinear.alpha_t(t)
    sigma_t = model._sigma_from_alphat(alpha_t.unsqueeze(-1))
    sigma_t = model._process_sigma(sigma_t)
    return model.backbone(z_t, sigma_t)

  @staticmethod
  def _expand_log_snr_to_probs(log_snr: torch.Tensor, prs: torch.Tensor) -> torch.Tensor:
    token_shape = prs.shape[:-1]
    if log_snr.shape == token_shape:
      return log_snr
    if (log_snr.ndim == 1 and len(token_shape) == 2
        and log_snr.shape[0] == token_shape[0]):
      return log_snr[:, None].expand(token_shape)
    if (log_snr.ndim == 2 and len(token_shape) == 2
        and log_snr.shape[0] == token_shape[0] and log_snr.shape[1] == 1):
      return log_snr.expand(token_shape)
    raise ValueError(
      f"log_snr shape {tuple(log_snr.shape)} is incompatible with "
      f"probability shape {tuple(prs.shape)}")

  @staticmethod
  def _marginal_probs_easydel(model, log_snr: torch.Tensor, prs: torch.Tensor) -> torch.Tensor:
    log_snr = GIDDSampler._expand_log_snr_to_probs(log_snr, prs)
    alpha = model.hybrid_noise.alpha_from_log_snr(log_snr)
    pi = model.hybrid_noise.pi_lambda(log_snr).to(dtype=prs.dtype)
    return alpha[..., None] * prs + (1 - alpha)[..., None] * pi

  def compute_posterior(self, model, z_t, t, s):
    """Compute posterior q(z_s | z_t, x_0) for GIDD.
    
    Args:
      model: The GIDD model instance.
      z_t: Current noisy samples at time t.
      t: Current timestep.
      s: Next (less noisy) timestep.
    
    Returns:
      Samples from the posterior distribution.
    """
    # Get model prediction (logits)
    logits = self._model_logits(model, z_t, t)

    logits = logits.clone()
    logits[..., model.mask_id] = model.neg_infinity

    pi = getattr(model.hybrid_noise, 'pi', None)
    supports_mask = True
    if pi is not None:
      supports_mask = bool((pi[model.mask_id] > 0).item())

    if supports_mask:
      probs = logits.softmax(-1)

      # Get noise schedule values
      if self._is_easydel_schedule(model):
        log_snr_s = model.hybrid_noise.log_snr_from_time(s)
        log_snr_t = model.hybrid_noise.log_snr_from_time(t)
        q_s = self._marginal_probs_easydel(model, log_snr_s, probs)
        q_t = self._marginal_probs_easydel(model, log_snr_t, probs)
      else:
        q_s = model.hybrid_noise.probs_at_t(probs, s)
        q_t = model.hybrid_noise.probs_at_t(probs, t)
      q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

      alpha_t, beta_pi_t = model.hybrid_noise.get_alpha_betapi(t)
      alpha_s, beta_pi_s = model.hybrid_noise.get_alpha_betapi(s)

      # Compute transition probabilities
      alpha_ts = alpha_t / alpha_s
      beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

      vz_t = F.one_hot(z_t, num_classes=model.vocab_size)
      beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(
        -1, z_t.unsqueeze(-1))
      q_ts = (alpha_ts.unsqueeze(1) * vz_t + beta_pi_ts_at_zt)
    else:
      mask_id = model.mask_id
      logits_nm = torch.cat([logits[..., :mask_id], logits[..., mask_id + 1:]], dim=-1)
      probs = logits_nm.softmax(-1)

      alpha_t, beta_pi_t = model.hybrid_noise.get_alpha_betapi(t)
      alpha_s, beta_pi_s = model.hybrid_noise.get_alpha_betapi(s)
      beta_pi_t = torch.cat([beta_pi_t[..., :mask_id], beta_pi_t[..., mask_id + 1:]], dim=-1)
      beta_pi_s = torch.cat([beta_pi_s[..., :mask_id], beta_pi_s[..., mask_id + 1:]], dim=-1)

      q_s = probs.mul(alpha_s.unsqueeze(-1))
      q_s[..., :beta_pi_s.shape[-1]].add_(beta_pi_s.unsqueeze(1))
      q_t = probs.mul(alpha_t.unsqueeze(-1))
      q_t[..., :beta_pi_t.shape[-1]].add_(beta_pi_t.unsqueeze(1))

      z_t_nm = z_t - (z_t > mask_id).to(z_t.dtype)
      q_zt = q_t.gather(-1, z_t_nm.unsqueeze(-1))

      # Compute transition probabilities
      alpha_ts = alpha_t / alpha_s
      beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

      vz_t = F.one_hot(z_t_nm, num_classes=model.vocab_size - 1)
      beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(
        -1, z_t_nm.unsqueeze(-1))
      q_ts = (alpha_ts.unsqueeze(1) * vz_t + beta_pi_ts_at_zt)

    # Compute posterior
    q_st = q_ts * q_s / q_zt
    
    # Optional: apply minimum probability threshold
    min_p = getattr(self.config.sampling, 'min_p', 0.0)
    if min_p > 0.0:
      is_small = (q_st < min_p).float()
      q_st = (1 - is_small) * q_st
      q_st = q_st / q_st.sum(-1, keepdim=True)
    
    return sample_categorical(q_st)

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Generate samples using GIDD reverse diffusion process.
    
    Args:
      model: The GIDD model instance.
      num_samples: Number of samples to generate.
      num_steps: Number of denoising steps.
      eps: Minimum timestep value (epsilon).
      inject_bos: Whether to inject BOS token at position 0.
    
    Returns:
      Generated token sequences of shape [num_samples, num_tokens].
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps
    if eps is None:
      eps = getattr(self.config.algo, 't_eps', 1e-4)

    # Sample from the prior (fully masked)
    z_t = model.hybrid_noise.sample_prior(
      (num_samples, model.num_tokens))
    if inject_bos:
      z_t[:, 0] = model.tokenizer.bos_token_id
    
    # Create timestep schedule from 1-eps to eps
    timesteps = torch.linspace(
      1 - eps, eps, num_steps + 1, device=model.device)
    
    # Iteratively denoise
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        num_samples, device=model.device)
      s = timesteps[i + 1] * torch.ones(
        num_samples, device=model.device)
      
      z_t = self.compute_posterior(model, z_t, t, s)
    
    return z_t
