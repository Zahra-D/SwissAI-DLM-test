"""UNI-D² Library

A modular framework for discrete diffusion models including MDLM, SEDD, UDLM, 
BD3LM, GIDD, and more. Built on PyTorch Lightning with Hydra configuration.

Main Components:
    - algorithms: Discrete diffusion algorithm implementations
    - models: Neural network backbones (DiT, BlockDiT, GPT-2 wrappers)
    - noise_schedules: Time-dependent noise schedules
    - forward_process: Forward diffusion processes (masking, uniform, etc.)
    - sampling: Iterative denoising samplers
    - data: Dataset loaders and tokenizers
    - training_objectives: Loss computation mixins

Entry Points:
    - CLI: python -m discrete_diffusion [config overrides]
    - API: from discrete_diffusion.train import train

Example:
    >>> from discrete_diffusion.train import train
    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.load('configs/config.yaml')
    >>> train(config)

See docs/00_architecture_overview.md for detailed architecture documentation.
"""
