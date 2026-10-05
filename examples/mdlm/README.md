# MDLM: Simple and Effective Masked Diffusion Language Models

Reference: [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)

MDLM simplifies masked diffusion for language modeling by removing complex reparameterizations. It uses a standard forward process that independently masks tokens and a reverse process that directly predicts the unmasked tokens. This approach achieves state-of-the-art perplexity among diffusion models, competitive with autoregressive baselines.

## Usage

Train on OpenWebText:

```bash
bash examples/mdlm/owt.sh
```

## Citation

```bibtex
@misc{sahoo2024simple,
      title={Simple and Effective Masked Diffusion Language Models}, 
      author={Subham Sekhar Sahoo and Marianne Arriola and Yair Schiff and Aaron Gokaslan and Edgar Marroquin and Justin T Chiu and Alexander Rush and Volodymyr Kuleshov},
      year={2024},
      eprint={2406.07524},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
