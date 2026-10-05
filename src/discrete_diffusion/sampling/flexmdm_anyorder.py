"""FlexMDM Any-Order Sampler.

This module implements the Euler sampling procedure for FlexMDM's any-order
mask insertion flow, which jointly samples token content and sequence length.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Sampler


def _sample_tokens(probs: torch.Tensor) -> torch.Tensor:
  """Sample one token per position from probability distribution.
  
  Args:
    probs: [batch_size, seq_len, vocab_size] transition probabilities
    
  Returns:
    [batch_size, seq_len] sampled token indices
  """
  batch_size, seq_len, vocab_size = probs.shape
  flat_probs = probs.view(-1, vocab_size)
  samples = torch.multinomial(flat_probs, num_samples=1)
  return samples.view(batch_size, seq_len)


class FlexMDMAnyOrderSampler(Sampler):
  """Sampler for FlexMDM any-order mask insertion flow.
  
  This implements Euler sampling for a joint process that both unmasks
  tokens and inserts new tokens (changing sequence length).
  """

  def __init__(self, config, forward_process=None):
    self.config = config

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Generate samples using Euler sampling.
    
    Args:
      model: FlexMDM model with interpolant
      num_samples: Number of samples to generate
      num_steps: Number of sampling steps
      eps: Small constant (unused, kept for interface compatibility)
      inject_bos: Whether to inject BOS token at position 0.

    Returns:
      Generated token sequences [num_samples, max_length]
    """
    del eps  # Unused

    device = model.device
    max_length = model.num_tokens
    batch_size = num_samples
    
    # Special tokens
    mask_token = model.tokenizer.mask_token_id
    pad_token = model.tokenizer.pad_token_id
    
    # Initialize with all padding
    xt = torch.full(
      (batch_size, max_length), pad_token, dtype=torch.int64, device=device
    )
    if inject_bos:
      xt[:, 0] = model.tokenizer.bos_token_id

    dt = 1.0 / num_steps
    t = torch.zeros(batch_size, device=device)
    
    # Precompute indices for scatter operations
    batch_idx_L = (
      torch.arange(batch_size, device=device)
      .view(batch_size, 1)
      .expand(batch_size, max_length)
    )
    pos_idx_L = (
      torch.arange(max_length, device=device)
      .view(1, max_length)
      .expand(batch_size, max_length)
    )
    
    for step_idx in range(num_steps):
      # Get model predictions
      prediction = model.forward(xt, t)
      pred_rate = model.interpolant.to_actual_rate(xt, prediction, t)
      unmask_rate = pred_rate.unmask_rate  # [B, L, V]
      len_rate = pred_rate.length_rate  # [B, L+1]
      
      # ——— Unmask step (Euler) ———
      mask_pos = (xt == mask_token).nonzero(as_tuple=True)
      
      # Zero out rates for non-masked positions
      unmask_rate[xt != mask_token] = 0
      # Zero out mask-to-mask transitions at masked positions
      unmask_rate[*mask_pos, mask_token] = 0
      # Set diagonal (stay at mask) to make rows sum to 0
      unmask_rate[*mask_pos, mask_token] = -unmask_rate[*mask_pos, :].sum(
        dim=1
      )
      
      # Convert rates to transition probabilities
      trans_prob = (unmask_rate * dt).clamp(0.0, 1.0)
      
      # Add "stay" probability for current tokens
      _xt = xt.clone()
      _xt[xt == pad_token] = mask_token
      trans_prob.scatter_add_(
        2,
        _xt.unsqueeze(-1),
        torch.ones_like(_xt.unsqueeze(-1), dtype=trans_prob.dtype),
      )
      
      # On final step, remove mask token from sampling
      if step_idx == num_steps - 1:
        trans_prob[*mask_pos, mask_token] = 0.0
      
      # Sample new tokens
      new_xt = _sample_tokens(trans_prob)
      new_xt[xt == pad_token] = pad_token
      new_xt = torch.where((xt != mask_token) & (xt != pad_token), xt, new_xt)
      
      # ——— Insertion step (only if not final step) ———
      if step_idx != num_steps - 1:
        # Sample number of tokens to insert at each gap
        ext = torch.bernoulli((len_rate * dt).clamp(0.0, 1.0)).long()  # [B, L+1]
        
        xt_len = xt.ne(pad_token).sum(dim=1)  # [B]
        
        # Only insert at valid gaps (before current length)
        gaps = torch.arange(max_length + 1, device=device).view(1, -1)
        ext = ext * (gaps <= xt_len.view(batch_size, 1)).long()
        
        total_ext = ext.sum(dim=1)
        
        # Check if insertion would exceed max_length
        valid = xt_len + total_ext <= max_length
        ext = ext * valid.view(batch_size, 1).long()
        
        # Compute cumulative extensions
        ext_ex = ext.int().cumsum(dim=1)  # [B, L+1]
        new_len = xt_len + total_ext  # [B]
        
        # Create new sequence with insertions
        xt_tmp = torch.full_like(xt, pad_token)
        mask_fill = pos_idx_L < new_len.view(batch_size, 1)
        xt_tmp[mask_fill] = mask_token
        
        # Scatter original tokens to new positions
        new_pos_orig = pos_idx_L + ext_ex[:, :max_length]  # [B, L]
        orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
        flat_b = batch_idx_L[orig_mask]
        flat_p = new_pos_orig[orig_mask]
        xt_tmp[flat_b, flat_p] = new_xt[orig_mask]
      else:
        # Final step: no insertion
        xt_tmp = new_xt
      
      xt = xt_tmp
      t = t + dt
    
    return xt


__all__ = ['FlexMDMAnyOrderSampler']

