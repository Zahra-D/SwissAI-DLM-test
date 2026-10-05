# Guide: Implementing a Custom Noise Schedule

This guide explains how to define a custom **Noise Schedule**. The noise schedule determines the rate at which information is destroyed during the forward process.

## Overview

A noise schedule provides two key functions of time $t \in [0, 1]$:
1. $\alpha(t)$: The signal retention probability.
   - $\alpha(0) = 1$ (Clean data)
   - $\alpha(1) \approx 0$ (Pure noise)
2. $\alpha'(t)$: The time derivative $\frac{d\alpha}{dt}$.
   - Used for weighting the training loss in continuous-time diffusion.

## Step 1: Create the Class

Create a new file `src/discrete_diffusion/noise_schedules/polynomial.py`. Subclass `NoiseSchedule`.

```python
import torch
from discrete_diffusion.noise_schedules.base import NoiseSchedule

class PolynomialNoiseSchedule(NoiseSchedule):
    """
    A polynomial decay schedule:
    alpha(t) = (1 - t)^power
    
    We add a small epsilon to prevent numerical instability at t=1.
    """

    def __init__(self, power: float = 2.0, eps: float = 1e-3):
        super().__init__()
        self.power = float(power)
        self.eps = float(eps)

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha(t).
        
        Formula: alpha(t) = (1 - t)^p * (1 - eps) + eps
        This ensures alpha(0)=1 and alpha(1)=eps.
        """
        # Ensure t is in [0, 1] if needed, though usually handled by caller
        term = (1 - t).pow(self.power)
        return term * (1 - self.eps) + self.eps

    def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute d/dt alpha(t).
        
        d/dt [ (1-t)^p * (1-eps) + eps ]
        = (1-eps) * p * (1-t)^(p-1) * (-1)
        = -p * (1-eps) * (1-t)^(p-1)
        """
        coeff = -self.power * (1 - self.eps)
        term = (1 - t).pow(self.power - 1)
        return coeff * term
```

## Step 2: Register the Schedule

Open `src/discrete_diffusion/noise_schedules/registry.py` and add your new class.

```python
# src/discrete_diffusion/noise_schedules/registry.py
from .polynomial import PolynomialNoiseSchedule

def build_noise_schedule(config, tokenizer=None):
    # ... existing code ...
    
    name = config.algo.noise_schedule.name
    params = config.algo.noise_schedule
    
    if name == 'polynomial':
        return PolynomialNoiseSchedule(
            power=params.get('power', 2.0),
            eps=params.get('eps', 1e-3)
        )
    
    # ... existing code ...
```

## Step 3: Configure

Update your config to use the new schedule.

```yaml
# configs/algo/my_custom_algo.yaml

noise_schedule:
  name: polynomial
  power: 3.0
  eps: 1e-4
```

