# SEDD: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

Reference: [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)

SEDD bridges the gap between score matching and discrete data by proposing score entropy, a novel loss that extends score principles to discrete spaces. This method allowed discrete diffusion models to beat existing paradigms and compete with autoregressive models.

## Usage

Train on OpenWebText:

```bash
bash examples/sedd/owt.sh
```

## Citation

```bibtex
@misc{lou2023discrete,
      title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution}, 
      author={Aaron Lou and Chenlin Meng and Stefano Ermon},
      year={2023},
      eprint={2310.16834},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
