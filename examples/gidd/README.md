# Generalized Interpolating Discrete Diffusion (GIDD)

Reference: [arXiv:2503.04482](https://arxiv.org/abs/2503.04482)

GIDD generalizes masked diffusion by deriving a new family of interpolating discrete diffusion processes that offer greater flexibility in designing noising processes. By leveraging a novel diffusion ELBO and combining masking with uniform noise, it enables the model to correct its own mistakes and improves sample quality.

## Usage

Train on OpenWebText:

```bash
bash examples/gidd/owt.sh
```

## Citation

```bibtex
@misc{rutte2025generalized,
      title={Generalized Interpolating Discrete Diffusion}, 
      author={Dimitri von Rütte and Janis Fluri and Yuhui Ding and Antonio Orvieto and Bernhard Schölkopf and Thomas Hofmann},
      year={2025},
      eprint={2503.04482},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
