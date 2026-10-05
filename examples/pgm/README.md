# PGM: Partition Generative Modeling

Reference: [arXiv:2505.18883](https://arxiv.org/abs/2505.18883)

PGM combines the strengths of autoregressive and masked generative models by partitioning tokens into two groups and using sparse attention to block information flow between them. This allows the model to process previously generated tokens only during sampling while retaining parallel and any-order generation capabilities, leading to significant improvements in sampling latency and throughput.

## Usage

Train on OpenWebText:

```bash
bash examples/pgm/owt.sh
```

## Citation

```bibtex
@misc{deschenaux2025partition,
      title={Partition Generative Modeling: Masked Modeling Without Masks}, 
      author={Justin Deschenaux and Lan Tran and Caglar Gulcehre},
      year={2025},
      eprint={2505.18883},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
