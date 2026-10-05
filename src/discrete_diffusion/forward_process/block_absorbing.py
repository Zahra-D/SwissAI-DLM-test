"""Block-absorbing forward process.

Minimal implementation: applies the absorbing process with a single shared
time `t` across the entire sequence (i.e., one block == whole sequence).
Returns `(masked_ids, p_mask, t_out)` where `t_out` is broadcast per-position.
"""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import _mask_token_id
from ..noise_schedules.base import NoiseSchedule


class BlockAbsorbingForwardProcess(ForwardProcess):
  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.mask_id = _mask_token_id(tokenizer)

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor | None = None):
    bsz, seqlen = input_ids.shape
    if t is None:
      # Sample a single t per sequence uniformly in [0,1]
      t = torch.rand(bsz, device=input_ids.device, dtype=torch.float32)
    alpha_t = self.schedule.alpha_t(t).view(-1, 1)
    p_mask = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = (torch.rand_like(input_ids, dtype=torch.float32) < p_mask).to(torch.bool)
    xt = torch.where(move_mask, torch.tensor(self.mask_id, device=input_ids.device, dtype=input_ids.dtype), input_ids)
    # Broadcast t per-position to satisfy per-block invariants
    t_out = t.view(-1, 1).expand(bsz, seqlen)
    return xt, p_mask, t_out


__all__ = ["BlockAbsorbingForwardProcess"]

