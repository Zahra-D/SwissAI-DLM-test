"""Hybrid Noising process for CANDI."""

from __future__ import annotations

import torch
from .utils import _mask_token_id
from .base import ForwardProcess
from ..noise_schedules.base import NoiseSchedule
import torch.nn.functional as F


class HybridForwardCANDI(ForwardProcess):
    """Hybrid noising kernel for CANDI.
    https://arxiv.org/pdf/2510.22510 

    Selects positions with probability `(1 - alpha_t)` to add Gaussian noise, with variance sigma_t.
    Leaves others the same.

    """

    def __init__(
        self, tokenizer, schedule: NoiseSchedule, name: str | None = None
    ) -> None:
        assert hasattr(schedule, "r_t") and hasattr(schedule, "sigma_t"), (
            "CANDI schedule must implement r_t and sigma_t methods."
        )

        super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
        self.mask_id = _mask_token_id(tokenizer)
        self.vocab_size = len(tokenizer)
        self._corruption_vocab_size = len(tokenizer) + 1

    def _discrete_noising(self, x, alpha_t):
        """Computes the noisy discrete sample."""
        p_mask = (1.0 - alpha_t).to(dtype=torch.float32)
        move_indices = torch.rand(*x.shape, device=x.device) < p_mask
        uniform_tensor = torch.randint(0, self._corruption_vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        return xt

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
        """Applies hybrid noising process to input_ids at time t. Implements Equation 11 of CANDI paper."""
        alpha_t = self.schedule.alpha_t(t).view(-1, 1)
        dalpha_t = self.schedule.alpha_prime_t(t).view(-1, 1)
        sigma_t = self.schedule.sigma_t(t).view(-1, 1, 1)

        disc_xt = self._discrete_noising(input_ids, alpha_t)
        reveal_mask = (disc_xt == input_ids).float()

        X_0 = F.one_hot(input_ids, num_classes=self.vocab_size).to(input_ids.device)
        X_t_prime = X_0 + torch.randn_like(X_0, dtype=torch.float32) * sigma_t
        X_t = X_0 * reveal_mask.unsqueeze(-1) + X_t_prime * (1 - reveal_mask).unsqueeze(-1)

        return {
            "xt": X_t, 
            "reveal_mask": reveal_mask, 
            "continuous_noise": sigma_t.squeeze(), 
            "discrete_noise": (1 - alpha_t).squeeze(),
            "alpha_t": alpha_t, 
            "dalpha_t": dalpha_t
        }  
