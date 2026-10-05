"""FlexMDM Joint Interpolant for Any-Order Mask Insertion Flow.

This module implements the forward process for FlexMDM's any-order algorithm,
which jointly models insertion (length) and masking (content) processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ForwardProcess
from .utils import _effective_vocab_size, _mask_token_id


class ScheduleProtocol(Protocol):
  def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    ...

  def sample_truncated(
    self, threshold: torch.Tensor, shape: Tuple[int, ...], device: torch.device
  ) -> torch.Tensor:
    ...

  def rate_scale_factor(self, t: torch.Tensor) -> torch.Tensor:
    ...


@dataclass
class ModelPrediction:
  """Model output for FlexMDM any-order algorithm.
  
  Attributes:
    token_logits: Logits for token predictions [B, L, V]
    length_posterior: Optional distribution over gap lengths [B, L, max_gap]
    expected_gaps: Expected number of tokens to insert [B, L]
  """
  token_logits: Tensor
  length_posterior: Optional[Tensor]
  expected_gaps: Tensor

  def __init__(
    self,
    token_logits: Tensor,
    length_posterior: Optional[Tensor] = None,
    expected_gaps: Optional[Tensor] = None,
  ):
    assert length_posterior is not None or expected_gaps is not None
    self.token_logits = token_logits
    self.length_posterior = length_posterior
    self.expected_gaps = expected_gaps
    if self.expected_gaps is None:
      _, _, L = self.length_posterior.shape
      index = torch.arange(0, L, device=token_logits.device).view(1, 1, -1)
      self.expected_gaps = (
        F.softmax(self.length_posterior, dim=-1) * index
      ).sum(dim=-1)


@dataclass
class Rate:
  """Rate information for sampling.
  
  Attributes:
    unmask_rate: Rate of unmasking transitions [B, L, V]
    length_rate: Rate of insertion transitions [B, L+1]
  """
  unmask_rate: Tensor  # Shape [Batch, Length, Vocab]
  length_rate: Tensor  # Shape [Batch, Length+1]


@dataclass
class JointInterpolantResult:
  """Result from sampling the joint interpolant.
  
  Attributes:
    xt: Noised sequence at time t [B, L]
    st: Sorting indices mapping xt back to x1 positions [B, L]
    _x1: Original clean sequence (stored for property computation)
    _pad_token: Padding token ID
    _mask_token: Mask token ID
  """
  xt: Tensor  # Shape [Batch, Length]
  st: Tensor  # Shape [Batch, Length]
  _x1: Tensor
  _pad_token: int
  _mask_token: int

  @property
  def mask_indices(self) -> Tensor:
    """Boolean mask indicating which positions are masked."""
    return self.xt == self._mask_token

  @property
  def unmasked(self) -> Tensor:
    """Ground truth tokens at positions corresponding to xt."""
    return torch.gather(self._x1, 1, self.st)

  @property
  def xt_length(self) -> Tensor:
    """Length of xt (excluding padding) [B]."""
    return (self.xt != self._pad_token).sum(dim=1)

  @property
  def x1_length(self) -> Tensor:
    """Length of x1 (excluding padding) [B]."""
    return (self._x1 != self._pad_token).sum(dim=1)

  @property
  def gaps_and_mask(self) -> tuple[Tensor, Tensor]:
    """Compute gap counts between xt positions.
    
    Returns:
      gaps: Number of deleted tokens between each position [B, L+1]
      mask: Valid positions mask [B, L+1]
    """
    x1_len = self.x1_length
    gaps = self.st.clone()

    # Add padding to compute differences
    pad_front = gaps.new_zeros((gaps.shape[0], 1)) - 1
    pad_back = gaps.new_zeros((gaps.shape[0], 1))
    gaps = torch.cat([pad_front, gaps, pad_back], dim=1)

    # Set the last gap to point to x1_len
    gaps.scatter_(
      1, self.xt_length.unsqueeze(1) + 1, x1_len.unsqueeze(1)
    )

    # Compute gaps as differences minus 1
    gaps = gaps[:, 1:] - gaps[:, :-1] - 1
    gaps = torch.clamp(gaps, min=0)

    # Create mask for valid positions
    idx = torch.arange(gaps.size(1), device=self.xt.device).unsqueeze(0)
    mask = idx <= self.xt_length.unsqueeze(1)
    gaps[~mask] = 0

    return gaps, mask


class FlexMDMForwardProcess(ForwardProcess):
  """Interpolant for any-order mask insertion flow.
  
  This implements a joint process where tokens are both inserted (affecting
  length) and masked (affecting content). The insertion and unmasking
  processes are governed by separate noise schedules.
  """

  def __init__(
    self,
    tokenizer,
    insertion_schedule: ScheduleProtocol,
    unmask_schedule: ScheduleProtocol,
    max_length: int,
    pad_token: int,
    name: str | None = None,
  ):
    """Initialize any-order interpolant.
    
    Args:
      tokenizer: Tokenizer instance
      insertion_schedule: Schedule for insertion process
      unmask_schedule: Schedule for unmasking process
      max_length: Maximum sequence length
      pad_token: ID of padding token
      name: Optional name for the process
    """
    super().__init__(tokenizer=tokenizer, schedule=None, name=name)
    self.insertion_schedule = insertion_schedule
    self.unmask_schedule = unmask_schedule
    self.max_length = max_length
    self.pad_token = pad_token
    self.mask_token = _mask_token_id(tokenizer)
    self.vocab_size = _effective_vocab_size(tokenizer)

  def hitting_time(self, t: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
    """Sample hitting times for insertion and unmasking.
    
    Insertion time is sampled uniformly, then unmasking time is sampled
    uniformly in [insertion_time, 1].
    
    Args:
      t: Current time [B]
      x1: Clean sequences [B, L]
      
    Returns:
      Tuple of (insertion_time [B, L], unmasking_time [B, L])
    """
    B, L = x1.shape
    eps = 1e-6

    insert_time = self.insertion_schedule.sample((B, L), device=x1.device)
    insert_time = eps + (1 - eps) * insert_time  # Ensure not exactly 0
    unmask_time = self.unmask_schedule.sample_truncated(
      insert_time, (B, L), device=x1.device
    )

    return insert_time, unmask_time

  def elbo_weight(self, t: Tensor, x1: Tensor):
    """Compute ELBO loss weights using rate scale factors.
    
    Args:
      t: Time values [B]
      x1: Clean sequences [B, L]
      
    Returns:
      Tuple of (unmask_weight [B, L], insert_weight [B, L+1])
    """
    insert_weight = self.insertion_schedule.rate_scale_factor(t)
    insert_weight = insert_weight[:, None].expand(-1, x1.shape[1] + 1)

    unmask_weight = self.unmask_schedule.rate_scale_factor(t)
    unmask_weight = unmask_weight.unsqueeze(1).expand(-1, x1.shape[1])

    return unmask_weight, insert_weight

  def to_actual_rate(
    self, xt: Tensor, prediction: ModelPrediction, t: Tensor
  ) -> Rate:
    """Convert model prediction to actual sampling rates.
    
    Args:
      xt: Current noised sequence [B, L]
      prediction: Model output
      t: Time values [B]
      
    Returns:
      Rate object with unmask and length rates
    """
    token_posterior = F.softmax(prediction.token_logits, dim=-1)  # [B, L, V]
    unmask_rate = token_posterior * self.unmask_schedule.rate_scale_factor(
      t
    ).view(-1, 1, 1)
    
    length_rate = (
      prediction.expected_gaps
      * self.insertion_schedule.rate_scale_factor(t).view(-1, 1)
    )

    return Rate(
      unmask_rate=unmask_rate,  # [B, L, V]
      length_rate=length_rate,  # [B, L+1]
    )

  def sample_interpolant(
    self, t: Tensor, x1: Tensor
  ) -> JointInterpolantResult:
    """Sample interpolant by applying deletion and masking.
    
    Tokens are deleted if t < insertion_time, masked if
    insertion_time <= t < unmasking_time, and clean otherwise.
    
    Args:
      t: Time values [B]
      x1: Clean sequences [B, L]
      
    Returns:
      JointInterpolantResult with noised sequence and metadata
    """
    # Sample hitting times for each token
    insertion_time, unmasking_time = self.hitting_time(t, x1)

    # Determine token states
    clean_tokens = x1.ne(self.pad_token)
    deleted_tokens = clean_tokens & (t[:, None] < insertion_time)
    masked_tokens = (
      clean_tokens
      & (t[:, None] >= insertion_time)
      & (t[:, None] < unmasking_time)
    )

    # Apply transformations
    xt = torch.where(
      deleted_tokens,
      self.pad_token,  # Deletion -> pad token
      torch.where(
        masked_tokens,
        self.mask_token,  # Masking -> mask token
        x1,  # Otherwise clean
      ),
    )

    # Sort to move padding to the end, track indices
    st = xt.ne(self.pad_token).argsort(dim=1, descending=True, stable=True)
    xt = torch.gather(xt, 1, st)
    st[xt == self.pad_token] = 0

    return JointInterpolantResult(
      xt=xt,
      st=st,
      _x1=x1,
      _pad_token=self.pad_token,
      _mask_token=self.mask_token,
    )

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
    """Return the noised tokens at time `t`.
    
    Returns:
      xt: Noised sequence [B, L]
      result: JointInterpolantResult containing metadata
    """
    result = self.sample_interpolant(t, input_ids)
    return result.xt, result


__all__ = [
  'ModelPrediction',
  'Rate',
  'JointInterpolantResult',
  'FlexMDMForwardProcess',
]


