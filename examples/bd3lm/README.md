# Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models

Reference: [arXiv:2503.09573](https://arxiv.org/abs/2503.09573)

Block Diffusion interpolates between discrete denoising diffusion and autoregressive models to support flexible-length generation and improve inference efficiency. It uses a block-based approach with KV caching and parallel token sampling, setting a new state-of-the-art among diffusion models on language modeling benchmarks.

## Usage

Train on OpenWebText:

```bash
bash examples/bd3lm/owt.sh
```

## Citation

```bibtex
@misc{arriola2025block,
      title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models}, 
      author={Marianne Arriola and Aaron Gokaslan and Justin T. Chiu and Zhihan Yang and Zhixuan Qi and Jiaqi Han and Subham Sekhar Sahoo and Volodymyr Kuleshov},
      year={2025},
      eprint={2503.09573},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
