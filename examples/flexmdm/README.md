# FlexMDM: Any-Order Flexible Length Masked Diffusion

Reference: [arXiv:2509.01025](https://arxiv.org/abs/2509.01025)

FlexMDM extends masked diffusion models to support flexible-length generation while retaining any-order inference capabilities. By inserting mask tokens and unmasking them using a stochastic interpolant framework, it models length statistics with high fidelity and achieves superior performance on tasks like math and code infilling compared to fixed-length baselines.

## Usage

Train on OpenWebText:

```bash
bash examples/flexmdm/owt.sh
```

## Citation

```bibtex
@misc{kim2025anyorder,
      title={Any-Order Flexible Length Masked Diffusion}, 
      author={Jaeyeon Kim and Lee Cheuk-Kit and Carles Domingo-Enrich and Yilun Du and Sham Kakade and Timothy Ngotiaoco and Sitan Chen and Michael Albergo},
      year={2025},
      eprint={2509.01025},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
