# Guide: Implementing a Custom Algorithm

This guide explains how to create a new discrete diffusion algorithm by subclassing the library's base classes.

## Overview

Algorithms in this library are **LightningModules** that orchestrate the training loop. They compose:
- A **Backbone Model** (predicts denoising)
- A **Noise Schedule** (defines $\alpha_t$)
- A **Forward Process** (defines $q(x_t | x_0)$)
- A **Sampler** (generates samples)

To add a custom algorithm, you typically need to:
1. Subclass `TrainerBase`, `Diffusion`, or `AbsorbingState`
2. Implement the loss computation logic (`nll_per_token`)
3. Implement output processing (`_process_model_output`)
4. Create a Hydra configuration file

## Step 1: Choose a Base Class

- **`TrainerBase`**: Minimal wrapper around PyTorch Lightning. Use for non-diffusion methods (like AR).
- **`Diffusion`**: Adds continuous-time diffusion logic (`_sample_t`, `compute_posterior`). Use for general diffusion.
- **`AbsorbingState`**: Specializes `Diffusion` for masking-based methods. Handles mask token management.

## Step 2: Implement the Algorithm

Create a new file `src/discrete_diffusion/algorithms/my_custom_algo.py`.

```python
import torch
import torch.nn.functional as F
from discrete_diffusion.algorithms.base import AbsorbingState

class MyCustomAlgo(AbsorbingState):
    """
    A custom discrete diffusion algorithm that implements a specific
    loss weighting or parameterization.
    """

    def __init__(self, config, tokenizer):
        # Initialize base class (sets up model, noise, forward_process)
        super().__init__(config, tokenizer)
        
        # Add custom hyperparameters
        self.custom_lambda = config.algo.get('lambda', 1.0)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        # Ensure compatible components
        assert self.config.algo.parameterization == 'subs'

    def _process_model_output(self, model_output, xt, sigma):
        """
        Convert raw model logits into log-probabilities.
        
        Args:
            model_output: [batch, seq_len, vocab_size]
            xt: [batch, seq_len]
            sigma: [batch]
        """
        # 1. Mask out invalid tokens (like the mask token itself if desired)
        model_output[:, :, self.mask_id] = self.neg_infinity

        # 2. Compute log_softmax
        log_probs = model_output.log_softmax(dim=-1)

        # 3. Enforce absorbing state constraint:
        #    If xt is NOT masked, probability mass must be concentrated on xt
        unmasked = (xt != self.mask_id)
        log_probs[unmasked] = self.neg_infinity
        log_probs[unmasked, xt[unmasked]] = 0.0

        return log_probs

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        """
        Compute the per-token negative log-likelihood (loss).

        Args:
            log_x_theta: Log probabilities from model [batch, seq_len, vocab]
            xt: Noisy input tokens [batch, seq_len]
            x0: Target clean tokens [batch, seq_len]
            alpha_t: Signal retention [batch, 1]
            dalpha_t: Derivative of alpha_t [batch, 1]
            low_var: Boolean flag for low-variance loss
        """
        # 1. Select log-prob of the ground truth token
        #    log_x_theta shape: [B, L, V] -> gather -> [B, L]
        log_p_theta = torch.gather(
            log_x_theta, 
            dim=-1, 
            index=x0.unsqueeze(-1)
        ).squeeze(-1)

        # 2. Compute Weighting
        if low_var:
            # Simple cross-entropy
            weight = -1.0 
        else:
            # Continuous-time diffusion weighting: alpha'(t) / (1 - alpha(t))
            # Note: dalpha_t is usually negative
            weight = dalpha_t / (1.0 - alpha_t + 1e-8)

        # 3. Apply custom logic (e.g., importance sampling, curriculum)
        loss = weight * log_p_theta * self.custom_lambda
        
        return loss
```

## Step 3: Create Configuration

Add a Hydra config file in `configs/algo/my_custom_algo.yaml`.

The `_target_` field tells Hydra which class to instantiate.

```yaml
# configs/algo/my_custom_algo.yaml
defaults:
  - /forward_process: absorbing
  - /noise_schedule: log_linear

# FULL CLASSPATH is critical here:
_target_: discrete_diffusion.algorithms.my_custom_algo.MyCustomAlgo

name: my_custom_algo
backbone: dit
parameterization: subs
time_conditioning: false

# Custom parameters
lambda: 1.5
```

## Step 4: Run Training

You can now run your algorithm by specifying the `algo` config group:

```bash
python -m discrete_diffusion \
    algo=my_custom_algo \
    data=text8 \
    model=small \
    trainer.max_steps=1000
```

## Key Methods to Override

| Method | Purpose |
|--------|---------|
| `__init__` | Custom setup. Always call `super().__init__`. |
| `_process_model_output` | **Required**. Transform raw logits to valid log-probs. |
| `nll_per_token` | **Required**. Define the loss function. |
| `generate_samples` | **Optional**. Implement custom sampling if the generic sampler isn't sufficient. |

## Common Pitfalls

1. **Vocab Size**: `AbsorbingState` may add a mask token. Ensure your model handles `vocab_size + 1`.
2. **NaN Losses**: Check `nll_per_token` denominators (`1 - alpha_t`).
3. **Hydra Target**: Ensure `_target_` path matches your directory structure exactly.

