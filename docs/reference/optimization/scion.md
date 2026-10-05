# SCION Optimizer

SCION constrains weight matrices via spectral normalization. Because activation functions introduce gain that spectral norm alone does not control, every nonlinearity must be wrapped with `activation_scale`.

## Activation scale values

| Activation | `activation_scale` |
|------------|--------------------|
| ReLU       | √2                 |
| GELU       | √2                 |
| SiLU       | √2                 |
| ReLU²      | 2                  |

The default is `1.0`, which is correct for standard AdamW training.

## Usage

```bash
python -m discrete_diffusion optim=scion model.activation_scale='${sqrt:2}'
```

For GIDD (ReLU²):

```bash
python -m discrete_diffusion optim=scion model.activation_scale=2.0
```
