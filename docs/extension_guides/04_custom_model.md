# Guide: Implementing a Custom Model (Backbone)

This guide explains how to add a new neural network architecture (backbone) to the library.

## Overview

The backbone model predicts the denoising distribution. It takes the noisy tokens $x_t$ and the noise level $\sigma_t$ (or time $t$) as input, and outputs logits over the vocabulary.

## Step 1: Define the Interface

Your model class must implement the following `forward` signature:

```python
def forward(self, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x_t: [batch, seq_len] LongTensor of token IDs
        sigma: [batch] FloatTensor of noise levels (or None if not time-conditioned)
        
    Returns:
        logits: [batch, seq_len, vocab_size] FloatTensor
    """
```

## Step 2: Implement the Model

Create a new file `src/discrete_diffusion/models/my_transformer.py`.

```python
import torch
import torch.nn as nn

class MyTransformer(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        dim = config.model.hidden_size
        
        # 1. Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.model.length, dim))
        
        # 2. Time Conditioning (Optional)
        self.time_conditioning = config.algo.time_conditioning
        if self.time_conditioning:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

        # 3. Main Body (e.g., PyTorch Transformer)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=config.model.n_heads,
            dim_feedforward=dim * 4,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.model.n_blocks)
        
        # 4. Output Projection
        self.head = nn.Linear(dim, vocab_size)
        
        # 5. Weight Tying (Optional but recommended)
        if config.model.get('tie_word_embeddings', False):
            self.head.weight = self.token_emb.weight

    def forward(self, x_t, sigma):
        # Embed tokens
        x = self.token_emb(x_t) + self.pos_emb[:, :x_t.size(1), :]
        
        # Inject time info
        if self.time_conditioning and sigma is not None:
            # Project sigma to embedding dimension
            t_emb = self.time_mlp(sigma.view(-1, 1)) # [B, D]
            x = x + t_emb.unsqueeze(1) # Add to all tokens
            
        # Process
        x = self.encoder(x)
        
        # Project to logits
        logits = self.head(x)
        return logits
```

## Step 3: Register the Model

Open `src/discrete_diffusion/models/registry.py` and add your builder function.

```python
# src/discrete_diffusion/models/registry.py
from .my_transformer import MyTransformer

def _build_my_transformer(config, vocab_size):
    return MyTransformer(config, vocab_size)

_BACKBONE_BUILDERS = {
    # ...
    'my-transformer': _build_my_transformer,
}
```

## Step 4: Configure

Update your config to use the new backbone name.

```yaml
# configs/algo/my_algo.yaml
backbone: my-transformer

# configs/model/small.yaml
hidden_size: 512
n_heads: 8
n_blocks: 6
dropout: 0.1
length: 1024
tie_word_embeddings: true
```

## Advanced: Block/Partition Inputs

If your algorithm (like BD3LM) uses block indices, your model signature needs to accept them:

```python
def forward(self, x_t, group_idxs, sigma):
    # ... use group_idxs for block-wise attention ...
```

Ensure you update the registry builder to pass `config` appropriately if your model needs to know about blocking strategies.

