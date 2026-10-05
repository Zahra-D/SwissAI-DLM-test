# UDLM: Uniform Discrete Diffusion Language Model

Reference: [arXiv:2412.10193](https://arxiv.org/abs/2412.10193)

UDLM uses uniform noise corruption with a novel continuous-time variational lower bound, enabling state-of-the-art performance among uniform noising methods. The forward process corrupts tokens towards a uniform distribution, and the model learns to reverse this process by minimizing a continuous-time ELBO.

## Usage

Train on text8:

```bash
bash examples/udlm/text8.sh
```

## Citation

```bibtex
@misc{schiff2024uniform,
      title={Uniform Discrete Diffusion Language Model}, 
      author={Yair Schiff and others},
      year={2024},
      eprint={2412.10193},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
