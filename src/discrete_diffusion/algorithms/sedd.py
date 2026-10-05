"""SEDD algorithm implementation extracted from :mod:`algorithms.algo`."""

import numpy as np
import torch

from . import base as trainer_base


class SEDD(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization != 'sedd':
      raise ValueError(self.parameterization)
    if not self.time_conditioning:
      raise ValueError('SEDD requires time conditioning.')

  def _process_model_output(self, model_output, xt, sigma):
    if sigma.ndim == 1:
      sigma = sigma[:, None]
    sigma = sigma.to(model_output.dtype)
    # Match bd3lms implementation: compute esigm1_log in one step to avoid intermediate tensor
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # model_output shape: (batch_size, diffusion_model_input_length, vocab_size)
    model_output = model_output - esigm1_log.unsqueeze(-1) - torch.tensor(
      np.log(model_output.shape[-1] - 1),
      dtype=model_output.dtype,
      device=model_output.device
    )
    # The below scatter operation sets the log score for the input word to 0.
    model_output = torch.scatter(model_output, -1, xt[..., None], torch.zeros_like(model_output[..., :1]))
    return model_output

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Compute the SEDD score-entropy term per token.

    Robust to the forward process choice:
    - Absorbing/mask-based: changed covers xt==mask_id since mask!=x0.
    - SEDD/uniform-replace: changed = (xt != x0).
    """
    if sigma.ndim == 1:
      sigma = sigma[:, None]
    sigma = sigma.to(log_score.dtype)

    # Positions that changed under the forward process
    changed = (xt != x0)
    if not torch.any(changed):
      return torch.zeros_like(xt, dtype=log_score.dtype)

    expsig_minus_1 = torch.expm1(sigma).expand(-1, xt.shape[1])
    q_ratio = 1 / expsig_minus_1[changed]

    # Negative term: q_ratio * log_score(x0)
    target_tokens = x0[changed]
    neg_term = q_ratio * torch.gather(
      log_score[changed],
      -1,
      target_tokens[..., None]).squeeze(-1)

    # Positive term: sum over all tokens except the current token at xt.
    # log_score has the log-score for xt set to 0 (score_xt = 1).
    score = log_score[changed].exp()
    current_tokens = xt[changed]
    score_xt = torch.gather(score, -1, current_tokens[..., None]).squeeze(-1)
    pos_term = score.sum(dim=-1) - score_xt

    const = q_ratio * (q_ratio.log() - 1)
    raw_entropy = (pos_term - neg_term + const).to(log_score.dtype)
    entropy = torch.zeros_like(log_score[..., 0])
    entropy = torch.masked_scatter(entropy, changed, raw_entropy)
    return entropy

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    sigma = self._sigma_from_alphat(alpha_t)
    entropy = self._score_entropy(log_x_theta, sigma, xt, x0)
    dalpha = torch.as_tensor(
      dalpha_t, device=alpha_t.device, dtype=alpha_t.dtype)
    while dalpha.dim() < alpha_t.dim():
      dalpha = dalpha.unsqueeze(-1)
    dalpha = dalpha.expand_as(alpha_t)
    dsigma = - dalpha / alpha_t
    return dsigma * entropy
