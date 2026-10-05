"""Base interface for discrete forward processes.

Forward processes encapsulate tokenizer-specific details and apply a chosen
noise schedule to produce noised latent variables `z_t` (or `x_t`).

This module only defines the abstract interface; concrete implementations
will be introduced separately.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ..noise_schedules.base import NoiseSchedule


class ForwardProcess(torch.nn.Module):
  """Abstract base class for discrete forward noising dynamics.

  Implementations should use `self.tokenizer` and `self.schedule` to compute
  noised states for given inputs and timesteps.
  """

  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__()
    self.tokenizer = tokenizer
    self.schedule = schedule
    self.name = name

  def forward(self, input_ids: torch.Tensor, t: torch.Tensor):  # pragma: no cover - abstract method
    """Return the noised tokens at time `t`.

    Concrete classes may return additional tensors as needed (e.g.,
    per-position `t` for blockwise sampling).
    """
    raise NotImplementedError


__all__ = ["ForwardProcess"]

