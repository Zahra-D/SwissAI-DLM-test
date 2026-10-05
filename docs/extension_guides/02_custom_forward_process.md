# Guide: Implementing a Custom Forward Process

This guide explains how to create a custom **Forward Process**. The forward process defines how clean data $x_0$ is corrupted into noisy data $x_t$ during training.

## Overview

The forward process $q(x_t | x_0)$ operates on token sequences. It takes a clean batch of tokens and a batch of timesteps, and applies noise according to a **Noise Schedule**.

Common processes include:
- **Absorbing**: Replace tokens with `[MASK]`.
- **Uniform**: Replace tokens with random vocabulary items.
- **SEDD**: Score-Entropy Discrete Diffusion noise.

## Step 1: Create the Class

Create a new file `src/discrete_diffusion/forward_process/dropout_noise.py`. Subclass `ForwardProcess`.

```python
import torch
from discrete_diffusion.forward_process.base import ForwardProcess
from discrete_diffusion.noise_schedules.base import NoiseSchedule
from discrete_diffusion.forward_process.utils import _mask_token_id

class DropoutForwardProcess(ForwardProcess):
    """
    A custom forward process that acts like dropout:
    Tokens are replaced with a special token (or mask) with probability (1 - alpha_t).
    """

    def __init__(self, tokenizer, schedule: NoiseSchedule):
        super().__init__(tokenizer, schedule)
        # You can define a custom noise token, or reuse the mask token
        self.noise_token_id = _mask_token_id(tokenizer)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
        """
        Apply noise to input_ids based on timestep t.

        Args:
            input_ids: Clean token IDs [batch, seq_len]
            t: Timesteps [batch] in [0, 1]

        Returns:
            noised_ids: [batch, seq_len]
            (Optional) metadata: You can return extra info if your algorithm needs it
        """
        # 1. Get signal retention probability alpha_t
        #    Shape: [batch, 1]
        alpha_t = self.schedule.alpha_t(t).view(-1, 1)

        # 2. Determine noise probability
        p_noise = 1.0 - alpha_t

        # 3. Sample mask: 1 where we should add noise
        #    Use float comparison for probability
        rand_vals = torch.rand_like(input_ids, dtype=torch.float32)
        noise_mask = rand_vals < p_noise

        # 4. Apply noise
        #    Where mask is True, replace with noise_token_id
        noised_ids = torch.where(
            noise_mask,
            torch.tensor(self.noise_token_id, device=input_ids.device),
            input_ids
        )

        return noised_ids
```

## Step 2: Register the Process

Open `src/discrete_diffusion/forward_process/registry.py` and add your new class to the builder.

```python
# src/discrete_diffusion/forward_process/registry.py
from .dropout_noise import DropoutForwardProcess  # Import your class

def build_forward_process(config, tokenizer, schedule):
    # ... existing code ...
    
    name = config.algo.forward_process.name
    
    if name == 'dropout':
        return DropoutForwardProcess(tokenizer, schedule)
    
    # ... existing code ...
```

## Step 3: Configure

Update your algorithm config to use the new process.

```yaml
# configs/algo/my_custom_algo.yaml

forward_process:
  name: dropout
```

## Advanced: Block-Wise or Structured Noise

If your process is more complex (e.g., masking contiguous blocks), you can implement that logic in `forward`.

For example, `BlockAbsorbingForwardProcess` returns additional metadata (like per-token timesteps). If you do this, ensure your **Algorithm** and **Sampler** are designed to handle the extra return values.

```python
    def forward(self, input_ids, t):
        # ... complex logic ...
        return xt, mask_prob, per_token_t
```

