"""GIDD linear diffusion with a constant noise distribution π.

Implements an interpolating discrete diffusion process of the form:

  q_t(z | x) = (1 - t) * 1[z = x] + t * π(z),

where π is a fixed categorical distribution over the vocabulary.

This module mirrors the `HybridDiffusion` interface used by the GIDD trainer:
`probs_at_t`, `sample_zt`, `sample_prior`, and `get_alpha_betapi`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical


def _get(cfg: object, path: str, default):
  cur = cfg
  for part in path.split('.'):
    if cur is None:
      return default
    if isinstance(cur, dict):
      cur = cur.get(part, default)
    else:
      cur = getattr(cur, part, default)
  return cur if cur is not None else default


def sample_t(config, batch_size: int, eps: float | None = None, device=None):
  if eps is None:
    eps = _get(config, 'algo.t_eps', _get(config, 'model.t_eps', 1e-4))

  low_disc = bool(_get(config, 'training.low_discrepancy_sampling',
                       _get(config, 'algo.low_discrepancy_sampling', False)))
  if low_disc:
    t = torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)
    t = (t + torch.rand(1, device=device, dtype=torch.float32)).fmod(1.0)
  else:
    t = torch.rand(batch_size, device=device, dtype=torch.float32)

  t = (1 - 2 * eps) * t + eps
  return t


class GiddLinearNoise(nn.Module):
  """Linear interpolation between clean tokens and a fixed distribution π.

  Args:
    tokenizer: Tokenizer-like object with `mask_token_id` and `__len__`.
    p_uniform: Mixture weight for a uniform-noise component (over non-mask tokens).
      The remaining probability mass `1 - p_uniform` is assigned to the mask token.
  """

  def __init__(self, tokenizer, p_uniform: float = 0.0):
    super().__init__()
    self.tokenizer = tokenizer
    self.mask_id = int(tokenizer.mask_token_id)
    self.vocab_size = int(len(tokenizer))

    p_uniform = float(p_uniform)
    p_uniform = min(max(p_uniform, 0.0), 1.0)

    mask = torch.zeros(self.vocab_size)
    mask[self.mask_id] = 1.0
    self.register_buffer('mask', mask, persistent=False)

    unif = (1.0 - self.mask) / max(self.vocab_size - 1, 1)
    self.register_buffer('unif', unif, persistent=False)

    pi = (1.0 - p_uniform) * self.mask + p_uniform * self.unif
    pi = pi / pi.sum().clamp_min(torch.finfo(pi.dtype).eps)
    self.register_buffer('pi', pi, persistent=False)

  def gather_pi(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Return π(x) for each token id in `input_ids`."""
    return self.pi[input_ids]

  def get_alpha_betapi(self, t: torch.Tensor, eps: float = 1e-4):
    del eps
    t = t[:, None]
    alpha_t = 1.0 - t
    beta_pi = t * self.pi.to(dtype=t.dtype)
    return alpha_t, beta_pi

  def probs_at_t(self, prs: torch.Tensor, t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    orig_dtype = prs.dtype
    alpha_t, beta_pi = self.get_alpha_betapi(t, eps=eps)

    probs = prs.mul(alpha_t.unsqueeze(-1))
    probs[..., :beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))
    return probs.to(orig_dtype)

  def sample_zt(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
    probs = self.probs_at_t(x, t)
    z_t = sample_categorical(probs)
    return z_t

  @torch.no_grad()
  def sample_prior(self, shape, *, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
      device = self.pi.device
    shape = tuple(shape)
    pi = self.pi.to(device=device, dtype=torch.float32)
    probs = pi.view(*((1,) * len(shape)), -1).expand(*shape, -1)
    return sample_categorical(probs)


__all__ = ['GiddLinearNoise', 'sample_t']

