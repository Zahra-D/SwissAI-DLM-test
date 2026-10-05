# CANDI: Hybrid Discrete-Continuous Diffusion Models

Reference: [arXiv:2510.22510](https://arxiv.org/abs/2510.22510)

CANDI addresses a fundamental limitation of continuous diffusion on discrete data by introducing "token identifiability" as an analytical lens. The method identifies two corruption mechanisms—discrete identity corruption and continuous rank degradation—that scale differently with vocabulary size, creating temporal dissonance. CANDI decouples discrete and continuous corruption processes, enabling simultaneous learning of both conditional structure and continuous geometry.

## Usage

Train on OpenWebText:

```bash
bash examples/candi/owt.sh
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise.r_min` | 0.05 | Minimum rank percentile for noise schedule |
| `noise.r_max` | 0.25 | Maximum rank percentile for noise schedule |
| `model.mixed_coeff` | 0.5 | Coefficient for biasing between mask/substitution |
| `model.length` | 1024 | Sequence length |
| `optim.lr` | 3e-4 | Learning rate |
| `lr_scheduler` | constant_warmup | LR scheduler with warmup |

## Citation

```bibtex
@misc{pynadath2025candi,
      title={CANDI: Hybrid Discrete-Continuous Diffusion Models},
      author={Patrick Pynadath and Jiaxin Shi and Ruqi Zhang},
      year={2025},
      eprint={2510.22510},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
