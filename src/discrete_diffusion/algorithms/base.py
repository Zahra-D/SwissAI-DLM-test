import itertools
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from ..evaluations import Metrics
from ..models import create_ema
from ..optimizers import Scion, ScionTrace
from .. import utils
from ..utils.utils import get_logger

logger = get_logger(__name__)
import omegaconf
from ..forward_process.utils import _effective_vocab_size, _unsqueeze


def ensure_mask_token(tokenizer):
  """Return mask token id and vocab size, ensuring the tokenizer exposes the mask."""
  vocab_size = _effective_vocab_size(tokenizer)
  if getattr(tokenizer, 'mask_token', None) is None:
    mask_id = vocab_size
    vocab_size += 1
  else:
    mask_id = tokenizer.mask_token_id
    vocab_size = max(vocab_size, int(mask_id) + 1)
  if getattr(tokenizer, 'mask_token_id', None) is None:
    setattr(tokenizer, 'mask_token_id', int(mask_id))
  return int(mask_id), vocab_size

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  num_tokens: torch.FloatTensor
  aux_metrics: object = None


class TrainerBase(L.LightningModule):
  """Base Trainer class for discrete diffusion models.
  
  Handles initialization of backbone, noise schedule, and sampler, as well as
  Lightning hooks for training and validation loops.
  """
  def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer, vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.ignore_bos = getattr(self.config.algo, 'ignore_bos', False)
    self.loss_type = getattr(self.config.algo, 'loss_type', None)
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = self.config.sampling.predictor
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    self._sampler_cfg = self._resolve_sampler_config()
    self._sampler = None
    
    target = self.config.model._target_
    instantiate_config = omegaconf.OmegaConf.create({'_target_': target})
    self.backbone = hydra.utils.instantiate(
      instantiate_config,
      self.config,
      self.vocab_size,
      _recursive_=False
    )
    self.model = self.backbone
    activation_scale = getattr(self.config.model, 'activation_scale', 1.0)
    if activation_scale != 1.0:
      logger.info(
        f'[SCION] activation_scale={activation_scale:.4g} '
        f'(set model.activation_scale=1.0 for standard AdamW training)')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    # Noise schedule - use Hydra instantiation
    # HybridDiffusion needs tokenizer passed at runtime
    if hasattr(self.config.noise, '_target_') and 'HybridDiffusion' in self.config.noise._target_:
      self.noise = hydra.utils.instantiate(self.config.noise, tokenizer=self.tokenizer)
    else:
      self.noise = hydra.utils.instantiate(self.config.noise)

    self.metrics = Metrics()

    self._prepare_ema()
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    if config.neg_infinity_mode == 'large-finite':
      self.neg_infinity = -1000000.0
    elif config.neg_infinity_mode == 'true-inf':
      self.neg_infinity = -float('inf')
    else:
      raise ValueError(f"neg_infinity_mode must be 'large-finite' or 'true-inf', got '{config.neg_infinity_mode}'")
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._last_depth_log_step = -1
    self._train_start_global_step = 0
    self._train_start_monotonic = None
    self._first_train_batch_logged = False
    # Lightning may restore ``batch_idx`` to a nonzero value while checkpoints
    # are written only at optimizer-step boundaries.  Accumulation therefore
    # has to be indexed relative to the first batch of this trainer invocation,
    # not by absolute ``batch_idx % accumulate_grad_batches``.
    self._train_batch_idx_origin = None
    # Autocast dtype of the backbone forward.  The historical default, fp32,
    # disables bf16 autocast, so compute runs in the parameter dtype (bf16 with
    # the default model.torch_dtype=bf16).  With fp32 master weights
    # (model.torch_dtype=fp32), set training.forward_autocast_dtype=bf16 to keep
    # bf16 compute.
    self._forward_autocast_dtype = {
      'fp32': torch.float32, 'float32': torch.float32,
      'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
    }[str(omegaconf.OmegaConf.select(
      self.config, 'training.forward_autocast_dtype', default='fp32')).lower()]
    self._perf_enabled = bool(
      omegaconf.OmegaConf.select(self.config, 'perf.enabled', default=False))
    self._perf_log_every_n_steps = max(
      1,
      int(omegaconf.OmegaConf.select(
        self.config, 'perf.log_every_n_steps', default=1)))
    self._perf_sync_cuda_for_timing = bool(
      omegaconf.OmegaConf.select(
        self.config, 'perf.sync_cuda_for_timing', default=True))
    self._perf_last_step_end_time = None
    self._last_perf_log_step = -1
    self._perf_optimizer_step_start_time = None
    self._last_optimizer_step_time_s = None
    self._perf_model_params = None
    self._perf_trainable_params = None
    self._perf_dense_trainable_params = None
    self._perf_flops_per_token = None
    self._perf_peak_flops_per_gpu = None
    self._trace_train_iter = None

    # Boundary initialization must act on complete tensors. Apply it after all
    # trainable modules exist but before Lightning can DDP/FSDP-wrap or shard
    # them. A resumed checkpoint is restored later and replaces these values.
    optimizer_name = str(getattr(self.config.optim, 'name', 'adamw')).lower()
    if (optimizer_name == 'scion'
        and bool(getattr(self.config.optim, 'boundary_init', False))):
      self._apply_scion_boundary_initialization()

  def _sync_cuda_for_perf_timing(self):
    if not self._perf_sync_cuda_for_timing:
      return
    device = getattr(self, 'device', None)
    if (device is not None
        and getattr(device, 'type', None) == 'cuda'
        and torch.cuda.is_available()):
      torch.cuda.synchronize(device)

  def _iter_unique_parameters(self):
    seen = set()
    for module in (self.backbone, self.noise):
      for parameter in module.parameters():
        ident = id(parameter)
        if ident in seen:
          continue
        seen.add(ident)
        yield parameter

  def _embedding_parameter_ids(self):
    ids = set()
    for module in (self.backbone, self.noise):
      for submodule in module.modules():
        if isinstance(submodule, torch.nn.Embedding):
          for parameter in submodule.parameters(recurse=False):
            ids.add(id(parameter))
    return ids

  def _world_size(self):
    world_size = int(getattr(self.trainer, 'world_size', 0) or 0)
    if world_size > 0:
      return world_size
    num_nodes = int(getattr(self.trainer, 'num_nodes', 1) or 1)
    num_devices = int(getattr(self.trainer, 'num_devices', 1) or 1)
    return max(1, num_nodes * num_devices)

  def _estimate_flops_per_token(self):
    manual_flops = omegaconf.OmegaConf.select(
      self.config, 'perf.flops_per_token', default=None)
    if manual_flops is not None:
      return float(manual_flops)

    trainable_params = float(
      self._perf_dense_trainable_params or self._perf_trainable_params or 0)
    model_cfg = self.config.model
    layers = int(
      getattr(model_cfg, 'n_blocks',
              getattr(model_cfg, 'num_hidden_layers', 0)) or 0)
    hidden_size = int(getattr(model_cfg, 'hidden_size',
                              getattr(model_cfg, 'dim', 0)) or 0)
    num_heads = int(getattr(model_cfg, 'n_heads',
                            getattr(model_cfg, 'num_attention_heads', 0)) or 0)
    head_dim = int(getattr(model_cfg, 'head_dim', 0) or 0)
    if head_dim <= 0 and hidden_size > 0 and num_heads > 0:
      head_dim = hidden_size // num_heads
    block_size = int(
      omegaconf.OmegaConf.select(
        self.config, 'block_size', default=self.config.model.length))

    # PaLM/nanoGPT-style train FLOPs estimate. The 6N term covers parameter
    # matmuls for forward+backward; the attention term covers dense QK/AV work.
    parameter_flops = 6.0 * trainable_params
    attention_flops = 0.0
    if layers > 0 and num_heads > 0 and head_dim > 0 and block_size > 0:
      attention_flops = 12.0 * layers * num_heads * head_dim * block_size
    return parameter_flops + attention_flops

  def _prepare_perf_metadata(self):
    if not self._perf_enabled:
      return
    parameters = list(self._iter_unique_parameters())
    self._perf_model_params = int(sum(p.numel() for p in parameters))
    self._perf_trainable_params = int(
      sum(p.numel() for p in parameters if p.requires_grad))
    tie_word_embeddings = bool(
      getattr(self.config.model, 'tie_word_embeddings', False))
    embedding_ids = set() if tie_word_embeddings else self._embedding_parameter_ids()
    embedding_params = int(
      sum(p.numel() for p in parameters
          if p.requires_grad and id(p) in embedding_ids))
    self._perf_dense_trainable_params = max(
      0, self._perf_trainable_params - embedding_params)
    self._perf_flops_per_token = self._estimate_flops_per_token()
    peak_tflops = omegaconf.OmegaConf.select(
      self.config, 'perf.theoretical_peak_tflops_per_gpu', default=None)
    self._perf_peak_flops_per_gpu = (
      None if peak_tflops is None else float(peak_tflops) * 1e12)

  def _log_perf_static_metadata(self):
    if (not self._perf_enabled
        or not getattr(self.trainer, 'is_global_zero', True)
        or self.trainer.logger is None):
      return
    global_batch_size = int(
      omegaconf.OmegaConf.select(self.config, 'loader.global_batch_size'))
    block_size = int(
      omegaconf.OmegaConf.select(
        self.config, 'block_size', default=self.config.model.length))
    tokens_per_step = float(global_batch_size * block_size)
    flops_per_step = float(self._perf_flops_per_token or 0.0) * tokens_per_step

    metrics = {
      'perf_constants/model_params': float(self._perf_model_params or 0),
      'perf_constants/trainable_params': float(self._perf_trainable_params or 0),
      'perf_constants/dense_trainable_params_estimate': float(
        self._perf_dense_trainable_params or 0),
      'perf_constants/flops_per_token_estimate': float(
        self._perf_flops_per_token or 0.0),
      'perf_constants/flops_per_step_estimate': flops_per_step,
      'perf_constants/world_size': float(self._world_size()),
      'perf_constants/gpu_count': float(self._world_size()),
      'perf_constants/global_batch_size': float(global_batch_size),
      'perf_constants/micro_batch_size_per_gpu': float(
        self.config.loader.batch_size),
      'perf_constants/grad_accum_steps': float(
        self.trainer.accumulate_grad_batches),
      'perf_constants/seq_len': float(block_size),
    }
    if self._perf_peak_flops_per_gpu is not None:
      metrics['perf_constants/theoretical_peak_tflops_per_gpu'] = (
        self._perf_peak_flops_per_gpu / 1e12)
    self.trainer.logger.log_metrics(
      metrics, step=int(getattr(self.trainer, 'global_step', 0) or 0))

  def _log_perf_metrics(self):
    if not self._perf_enabled:
      return
    global_step = int(getattr(self.trainer, 'global_step', 0) or 0)
    if global_step <= 0 or global_step == self._last_perf_log_step:
      return

    self._sync_cuda_for_perf_timing()
    now = time.perf_counter()
    previous_end = self._perf_last_step_end_time
    self._perf_last_step_end_time = now
    self._last_perf_log_step = global_step

    if previous_end is None:
      return
    step_time_s = now - previous_end
    if step_time_s <= 0:
      return
    if global_step % self._perf_log_every_n_steps != 0:
      return
    if (not getattr(self.trainer, 'is_global_zero', True)
        or self.trainer.logger is None):
      return

    global_batch_size = int(
      omegaconf.OmegaConf.select(self.config, 'loader.global_batch_size'))
    block_size = int(
      omegaconf.OmegaConf.select(
        self.config, 'block_size', default=self.config.model.length))
    world_size = self._world_size()
    tokens_per_step = float(global_batch_size * block_size)
    tokens_per_s = tokens_per_step / step_time_s
    flops_per_step = float(self._perf_flops_per_token or 0.0) * tokens_per_step
    achieved_flops_per_s = flops_per_step / step_time_s
    achieved_flops_per_s_per_gpu = achieved_flops_per_s / max(world_size, 1)

    metrics = {
      'perf/step_time_s': step_time_s,
      'perf/tokens_per_s': tokens_per_s,
      'perf/tokens_per_s_per_gpu': tokens_per_s / max(world_size, 1),
      'perf/samples_per_s': float(global_batch_size) / step_time_s,
      'perf/optimizer_step_time_s': float(
        self._last_optimizer_step_time_s or 0.0),
      'perf/achieved_tflops': achieved_flops_per_s / 1e12,
      'perf/achieved_tflops_per_gpu': achieved_flops_per_s_per_gpu / 1e12,
    }
    if self._perf_peak_flops_per_gpu:
      metrics['perf/mfu'] = (
        achieved_flops_per_s_per_gpu / self._perf_peak_flops_per_gpu)

    device = getattr(self, 'device', None)
    if (device is not None
        and getattr(device, 'type', None) == 'cuda'
        and torch.cuda.is_available()):
      metrics.update({
        'perf/cuda_memory_allocated_gb':
          torch.cuda.memory_allocated(device) / 1e9,
        'perf/cuda_memory_reserved_gb':
          torch.cuda.memory_reserved(device) / 1e9,
        'perf/max_cuda_memory_allocated_gb':
          torch.cuda.max_memory_allocated(device) / 1e9,
        'perf/max_cuda_memory_reserved_gb':
          torch.cuda.max_memory_reserved(device) / 1e9,
      })

    self.trainer.logger.log_metrics(metrics, step=global_step)

  def _prepare_ema(self):
    if self.config.training.ema > 0:
      self.ema = create_ema(self._get_parameters(), decay=self.config.training.ema)
    else:
      self.ema = None

  def _validate_configuration(self):
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def q_xt(self, x, t):
    raise NotImplementedError
  
  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(), self.noise.parameters())

  def _named_trainable_parameters(self):
    """Yield (name, parameter) pairs across backbone + noise without duplicates."""
    seen = set()
    for module_name, module in (("backbone", self.backbone), ("noise", self.noise)):
      for name, param in module.named_parameters():
        if not param.requires_grad:
          continue
        pid = id(param)
        if pid in seen:
          continue
        seen.add(pid)
        yield f"{module_name}.{name}", param

  def _build_adamw_jax_style_param_groups(self):
    """Build AdamW param groups similar to EasyDeL JAX train.py grouping.

    Groups:
    - bulk_params: matrices/tensors with ndim > 1 (except explicit groups below)
    - ln_params: params with "norm" in their name
    - bias_params: params with "bias" in their name
    - emb_unemb_params: params with "embed_tokens" or "lm_head" in their name
    """
    optim_cfg = self.config.optim
    hidden_size = float(getattr(self.config.model, "hidden_size", 1))
    if hidden_size <= 0:
      raise ValueError(f"Expected positive hidden_size, got: {hidden_size}")

    base_lr = float(getattr(optim_cfg, "lr", 0.0))
    aux_lr_factor = float(getattr(optim_cfg, "aux_lr_factor", 0.02))
    bulk_divide_by_hidden = bool(getattr(optim_cfg, "bulk_lr_divide_by_hidden_size", True))

    bulk_lr = (base_lr / hidden_size) if bulk_divide_by_hidden else base_lr
    aux_lr = base_lr * aux_lr_factor

    weight_decay = float(getattr(optim_cfg, "weight_decay", 0.0))
    ln_wd = float(getattr(optim_cfg, "ln_wd", weight_decay))
    bias_wd = float(getattr(optim_cfg, "bias_wd", 0.0))
    bulk_wd = float(getattr(optim_cfg, "bulk_wd", weight_decay))
    emb_wd = float(getattr(optim_cfg, "emb_unemb_wd", 0.0))

    grouped = {
      "bulk_params": [],
      "ln_params": [],
      "bias_params": [],
      "emb_unemb_params": [],
    }

    for name, param in self._named_trainable_parameters():
      lname = name.lower()
      if "norm" in lname:
        grouped["ln_params"].append(param)
      elif "embed_tokens" in lname or "lm_head" in lname:
        grouped["emb_unemb_params"].append(param)
      elif "bias" in lname:
        grouped["bias_params"].append(param)
      elif param.ndim > 1:
        grouped["bulk_params"].append(param)
      else:
        grouped["bulk_params"].append(param)

    param_groups = []
    if grouped["bulk_params"]:
      param_groups.append({
        "params": grouped["bulk_params"],
        "name": "bulk_params",
        "lr": bulk_lr,
        "weight_decay": bulk_wd,
      })
    if grouped["ln_params"]:
      param_groups.append({
        "params": grouped["ln_params"],
        "name": "ln_params",
        "lr": aux_lr,
        "weight_decay": ln_wd,
      })
    if grouped["bias_params"]:
      param_groups.append({
        "params": grouped["bias_params"],
        "name": "bias_params",
        "lr": aux_lr,
        "weight_decay": bias_wd,
      })
    if grouped["emb_unemb_params"]:
      param_groups.append({
        "params": grouped["emb_unemb_params"],
        "name": "emb_unemb_params",
        "lr": aux_lr,
        "weight_decay": emb_wd,
      })

    if not param_groups:
      raise ValueError("No trainable parameters found for AdamW parameter groups.")

    return param_groups

  def _build_scion_optimizer_groups(self):
    """Build SCION groups with configurable learning-rate parameterization.

    Groups:
    - embedding: input/output embedding parameters (embed_tokens + lm_head)
    - bias: all non-embedding parameters whose name contains "bias" (any ndim)
    - one_d: all non-embedding, non-bias parameters with ndim <= 1 (norm vectors, scalars)
    - matrix: all remaining non-embedding parameters with ndim > 1
    """
    optim_cfg = self.config.optim
    base_lr = optim_cfg.lr
    hidden_size = self.config.model.hidden_size 
    aux_lr_factor = optim_cfg.aux_lr_factor

    equal_group_lr = bool(getattr(optim_cfg, 'equal_group_lr', False))
    if equal_group_lr:
      matrix_lr = base_lr
      aux_lr = base_lr
    else:
      matrix_lr = base_lr / hidden_size
      aux_lr = base_lr * aux_lr_factor


    scale_embed = optim_cfg.scale_embed
    scale_bias = optim_cfg.scale_bias
    scale_layer_norm = optim_cfg.scale_layer_norm
    scale_matrix = optim_cfg.scale_matrix
    
    spectral_steps = optim_cfg.spectral_norm_steps
    bias_norm = optim_cfg.bias_norm
    norm_layer_norm = optim_cfg.norm_layer_norm
    backbone = self.backbone
    emb_params = []
    bias_params = []
    layer_norm_params = []
    matrix_params = []
    emb_ids = set()

    def _collect_ids(module):
      if module is None:
        return
      for p in module.parameters():
        if p.requires_grad:
          emb_ids.add(id(p))

    # DIT:
    #   embedding group: self.output_layer + self.vocab_embed
    if (hasattr(backbone, 'blocks')
        and hasattr(backbone, 'vocab_embed')
        and hasattr(backbone, 'output_layer')):
      _collect_ids(backbone.output_layer)
      _collect_ids(backbone.vocab_embed)
    # GiddHFWrapper:
    #   embedding group: self.model.lm_head + self.model.model.embed_tokens
    elif (hasattr(backbone, 'model')
          and hasattr(backbone.model, 'model')
          and hasattr(backbone.model.model, 'layers')
          and hasattr(backbone.model.model, 'embed_tokens')):
      _collect_ids(getattr(backbone.model, 'lm_head', None))
      _collect_ids(backbone.model.model.embed_tokens)
    else:
      model_target = getattr(getattr(self.config, 'model', object()), '_target_', 'unknown')
      raise ValueError(
        'SCION grouping supports DIT and GiddHFWrapper backbones only '
        f"(got model target: {model_target}).")

    # Assign each trainable parameter exactly once (safe with tied/shared weights).
    # Keep names for validation + debug printing.
    emb_names = []
    bias_names = []
    layer_norm_names = []
    matrix_names = []
    for name, p in self._named_trainable_parameters():
      pid = id(p)
      lname = name.lower()
      if pid in emb_ids:
        emb_params.append(p)
        emb_names.append(f"{name} {tuple(p.shape)}")
      elif "bias" in lname:
        bias_params.append(p)
        bias_names.append(f"{name} {tuple(p.shape)}")
      elif p.ndim <= 1:
        layer_norm_params.append(p)
        layer_norm_names.append(f"{name} {tuple(p.shape)}")
      else:
        matrix_params.append(p)
        matrix_names.append(f"{name} {tuple(p.shape)}")

    if not emb_params:
      raise ValueError('SCION expected non-empty embedding parameters, but found none.')
    if not matrix_params:
      raise ValueError('SCION expected non-empty matrix parameters, but found none.')

    groups = []
    if emb_params:
      groups.append({
        'params': emb_params,
        'name': 'embedding',
        'norm': 'Sign',
        'norm_kwargs': {},
        'scale': scale_embed,
        'lr': aux_lr,
      })
    if bias_params:
      groups.append({
        'params': bias_params,
        'name': 'bias',
        'norm': bias_norm,
        'norm_kwargs': {},
        'scale': scale_bias,
        'lr': aux_lr,
      })
    if layer_norm_params:
      groups.append({
        'params': layer_norm_params,
        'name': 'one_d',
        'norm': norm_layer_norm,
        'norm_kwargs': {},
        'scale': scale_layer_norm,
        'lr': aux_lr,
      })
    if matrix_params:
      groups.append({
        'params': matrix_params,
        'name': 'matrix',
        'norm': 'Spectral',
        'norm_kwargs': {'steps': spectral_steps},
        'scale': scale_matrix,
        'lr': matrix_lr,
      })

    # Print explicit group membership once (rank 0) for verification.
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if global_rank == 0 and local_rank == 0:
      logger.info("[SCION] Grouping summary:")
      logger.info(
        "[SCION] LR parameterization: %s (FW stepsize=%g)",
        "equal across all groups" if equal_group_lr else "legacy scaled groups",
        float(base_lr))
      for group in groups:
        logger.info(
          "[SCION] group=%s norm=%s lr=%g radius=%g",
          group['name'], group['norm'], float(group['lr']),
          float(group['scale']))
      logger.info("[SCION] embedding: %d params", len(emb_names))
      for n in emb_names:
        logger.info("[SCION][embedding] %s", n)
      logger.info("[SCION] bias: %d params", len(bias_names))
      for n in bias_names:
        logger.info("[SCION][bias] %s", n)
      logger.info("[SCION] layer_norm: %d params", len(layer_norm_names))
      for n in layer_norm_names:
        logger.info("[SCION][layer_norm] %s", n)
      logger.info("[SCION] matrix: %d params", len(matrix_names))
      for n in matrix_names:
        logger.info("[SCION][matrix] %s", n)

    return groups

  def _apply_scion_boundary_initialization(self):
    """Initialize every SCION group on its configured LMO boundary.

    ``Scion.init`` delegates to the group's norm backend for the Section B.3
    draw (semi-orthogonal, normalized Gaussian, or random sign) and multiplies
    the result by the group's radius/``scale``. The dimension-dependent
    scaling inside each norm's own ``init`` (e.g. Spectral's sqrt(dout/din))
    always applies; only that final radius multiply is affected here.

    ``optim.boundary_init_scale``, if set, overrides the radius used for this
    initial draw only, decoupling it from the radius (``scale``) used by
    every subsequent ``Scion.step()`` LMO update. Leave unset to keep the
    original B.3 behavior (init radius == training radius).
    """
    groups = self._build_scion_optimizer_groups()
    init_scale_override = getattr(self.config.optim, 'boundary_init_scale', None)
    if init_scale_override is not None:
      init_groups = [dict(g, scale=float(init_scale_override)) for g in groups]
      logger.info(
        '[SCION] boundary_init_scale override active: initializing at radius '
        '%g instead of each group\'s training radius.', float(init_scale_override))
    else:
      init_groups = groups
    initializer = Scion(
      init_groups,
      lr=float(self.config.optim.lr),
      momentum=float(self.config.optim.momentum),
      unconstrained=bool(getattr(self.config.optim, 'unconstrained', False)),
    )
    initializer.init()
    logger.info(
      '[SCION] Applied Section B.3 boundary initialization to %d groups: %s',
      len(groups), ', '.join(group['name'] for group in groups))

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed']
    # is 1 iteration behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps,
    # not the number of local steps, so we don't multiply with
    # self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']

  def on_sanity_check_start(self):
    logger.info('[TRAIN_TIMING] on_sanity_check_start')

  def on_sanity_check_end(self):
    logger.info('[TRAIN_TIMING] on_sanity_check_end')

  def on_train_start(self):
    logger.info('[TRAIN_TIMING] on_train_start')
    # ``Metrics`` is a helper object rather than a registered ``nn.Module``.
    # Lightning/FSDP therefore does not reliably move its torchmetrics state
    # with the model.  Keeping the state on CPU makes torchmetrics attempt a
    # CPU all-gather on the NCCL process group at epoch end.
    self.metrics.to(self.device)
    self._train_start_global_step = int(
      getattr(self.trainer, 'global_step', 0) or 0)
    self._train_start_monotonic = time.monotonic()
    self._first_train_batch_logged = False
    self._train_batch_idx_origin = None
    if self._perf_enabled:
      self._prepare_perf_metadata()
      self._sync_cuda_for_perf_timing()
      self._perf_last_step_end_time = time.perf_counter()
      self._log_perf_static_metadata()
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)

  def on_before_optimizer_step(self, optimizer):
    if self._perf_enabled:
      self._sync_cuda_for_perf_timing()
      self._perf_optimizer_step_start_time = time.perf_counter()
    if hasattr(optimizer, "track_stats") and bool(
      getattr(self.config.optim, "trace_collect_noise_stats", False)
    ):
      self._collect_scion_trace_m_stats(optimizer)

  def _trace_batch_to_device(self, batch):
    if torch.is_tensor(batch):
      return batch.to(self.device, non_blocking=True)
    if isinstance(batch, dict):
      return {k: self._trace_batch_to_device(v) for k, v in batch.items()}
    if isinstance(batch, tuple):
      return tuple(self._trace_batch_to_device(v) for v in batch)
    if isinstance(batch, list):
      return [self._trace_batch_to_device(v) for v in batch]
    return batch

  def _next_trace_batch(self):
    trace_dataloader = getattr(self, '_scion_trace_dataloader', None)
    if trace_dataloader is None:
      # Compatibility fallback for older callers that did not construct a
      # dedicated trace loader. New rho/sigma runs always use the independent
      # loader above, so this path is not used with persistent workers.
      trace_dataloader = getattr(self.trainer, "train_dataloader", None)
    if trace_dataloader is None:
      return None
    # Rho/sigma probing uses an auxiliary iterator.  It must not advance the
    # checkpointed cursor of the optimizer-training iterator.
    sampler = getattr(trace_dataloader, "sampler", None)
    sampler_state = None
    if (hasattr(sampler, "state_dict")
        and hasattr(sampler, "load_state_dict")):
      sampler_state = sampler.state_dict()
    try:
      if self._trace_train_iter is None:
        self._trace_train_iter = iter(trace_dataloader)
      try:
        batch = next(self._trace_train_iter)
      except StopIteration:
        self._trace_train_iter = iter(trace_dataloader)
        batch = next(self._trace_train_iter)
    finally:
      if sampler_state is not None:
        sampler.load_state_dict(sampler_state)
    return self._trace_batch_to_device(batch)

  def _optimizer_params(self, optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]

  def _save_optimizer_grads(self, optimizer):
    return [
      None if p.grad is None else p.grad.detach().clone()
      for p in self._optimizer_params(optimizer)
    ]

  def _restore_optimizer_grads(self, optimizer, saved_grads):
    for p, grad in zip(self._optimizer_params(optimizer), saved_grads):
      p.grad = None if grad is None else grad.to(device=p.device)

  def _average_optimizer_grads(self, optimizer):
    if not (torch.distributed.is_available()
            and torch.distributed.is_initialized()):
      return
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
      return
    grads = [
      p.grad for p in self._optimizer_params(optimizer)
      if p.grad is not None]
    if not grads:
      return
    by_dtype = {}
    for g in grads:
      by_dtype.setdefault(g.dtype, []).append(g)
    # One all_reduce per dtype group instead of one per parameter tensor:
    # per-call collective overhead dominates for hundreds of small tensors.
    for tensors in by_dtype.values():
      flat = torch._utils._flatten_dense_tensors(tensors)
      torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
      flat.div_(world_size)
      for t, synced in zip(
          tensors, torch._utils._unflatten_dense_tensors(flat, tensors)):
        t.copy_(synced)

  def _collect_scion_trace_m_stats(self, optimizer):
    every = int(getattr(self.config.optim, "trace_noise_stats_every", 0))
    if every <= 0:
      return
    next_step = int(getattr(self.trainer, "global_step", 0) or 0) + 1
    if next_step % every != 0:
      return

    trace_m = int(getattr(self.config.optim, "trace_m", 64))
    min_samples = int(getattr(self.config.optim, "trace_noise_min_samples", 3))
    trace_m = max(trace_m, min_samples, 1)
    progress_every = max(0, int(getattr(
      self.config.optim, 'trace_progress_every', 0)))
    accumulation_steps = max(1, int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1))
    saved_grads = self._save_optimizer_grads(optimizer)
    was_training = self.training

    try:
      logger.info(
        '[SCION_TRACE] begin global_step=%d, trace_m=%d, global_batch=%d, '
        'accumulation=%d', next_step, trace_m,
        int(self.config.loader.global_batch_size), accumulation_steps)
      self.train()
      for trace_idx in range(trace_m):
        optimizer.zero_grad(set_to_none=True)
        self.zero_grad(set_to_none=True)
        sample_loss = None

        for accum_step in range(accumulation_steps):
          batch = self._next_trace_batch()
          if batch is None:
            return
          losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            current_accumulation_step=accum_step,
            train_mode=True,
          )
          # Accumulate microbatch sums here, then divide gradients and loss
          # exactly once after the loop.  Dividing here as well as below would
          # scale gradients by 1 / accumulation_steps**2 and corrupt sigma^2.
          loss = losses.loss
          sample_loss = loss.detach() if sample_loss is None else sample_loss + loss.detach()
          loss.backward()

        if accumulation_steps > 1:
          for p in self._optimizer_params(optimizer):
            if p.grad is not None:
              p.grad.div_(accumulation_steps)
          sample_loss = sample_loss / accumulation_steps

        self._average_optimizer_grads(optimizer)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
          sample_loss = sample_loss.clone()
          torch.distributed.all_reduce(sample_loss, op=torch.distributed.ReduceOp.SUM)
          sample_loss = sample_loss / torch.distributed.get_world_size()

        optimizer.track_stats(cur_loss=sample_loss, store_on_cpu=True)
        if (progress_every > 0
            and ((trace_idx + 1) % progress_every == 0
                 or trace_idx + 1 == trace_m)):
          logger.info(
            '[SCION_TRACE] collected %d/%d global-gradient samples',
            trace_idx + 1, trace_m)

      optimizer.report_stats()
      logger.info('[SCION_TRACE] report complete for global_step=%d', next_step)
    finally:
      optimizer.zero_grad(set_to_none=True)
      self.zero_grad(set_to_none=True)
      self._restore_optimizer_grads(optimizer, saved_grads)
      if not was_training:
        self.eval()

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if (self._perf_enabled
        and self._perf_optimizer_step_start_time is not None):
      self._sync_cuda_for_perf_timing()
      self._last_optimizer_step_time_s = (
        time.perf_counter() - self._perf_optimizer_step_start_time)
      self._perf_optimizer_step_start_time = None
    # Log per-parameter-group learning rates (useful with grouped AdamW configs).
    if getattr(self.trainer, "is_global_zero", True):
      for opt_idx, opt in enumerate(getattr(self.trainer, "optimizers", [])):
        for group_idx, group in enumerate(getattr(opt, "param_groups", [])):
          group_name = str(group.get("name", f"group_{group_idx}"))
          lr_val = float(group.get("lr", 0.0))
          self.log(
            name=f"trainer/lr/{opt_idx}/{group_name}",
            value=lr_val,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            prog_bar=False,
          )
    if self.ema: self.ema.update(self._get_parameters())
    if getattr(self.trainer, "is_global_zero", True):
      for opt in getattr(self.trainer, "optimizers", []):
        if hasattr(opt, "pop_trace_logs"):
          for key, value in opt.pop_trace_logs().items():
            self.log(
              name=key,
              value=float(value),
              on_step=True,
              on_epoch=False,
              sync_dist=False,
              prog_bar=False,
            )
        if hasattr(opt, "pop_noise_logs"):
          for key, value in opt.pop_noise_logs().items():
            self.log(
              name=key,
              value=float(value),
              on_step=True,
              on_epoch=False,
              sync_dist=False,
              prog_bar=False,
            )

  def configure_gradient_clipping(
      self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
    """Gradient clipping that works under both DDP and FSDP.

    DDP: every rank already holds the full, all-reduced gradient for every
    parameter, so Lightning's default clip_grad_norm_-based path is already
    correct as-is -- delegate to it unchanged.

    FSDP: gradients are sharded across ranks, so a plain
    torch.nn.utils.clip_grad_norm_ over local shards would compute a
    per-shard norm, not the true global norm, and silently under-clip.
    FSDP's own model.clip_grad_norm_() performs the required cross-rank
    all-reduce internally, so it must be called on the FSDP-wrapped root
    module instead of going through Lightning's default path.
    """
    strategy = getattr(self.trainer, 'strategy', None)
    is_fsdp = isinstance(strategy, L.pytorch.strategies.FSDPStrategy)

    if not is_fsdp:
      super().configure_gradient_clipping(
        optimizer,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm)
      return

    clip_val = (
      gradient_clip_val if gradient_clip_val is not None
      else self.trainer.gradient_clip_val)
    if clip_val is None or float(clip_val) <= 0:
      return

    algorithm = (
      gradient_clip_algorithm if gradient_clip_algorithm is not None
      else self.trainer.gradient_clip_algorithm)
    algorithm = str(getattr(algorithm, 'value', algorithm) or 'norm').lower()
    if algorithm != 'norm':
      raise NotImplementedError(
        f"FSDP gradient clipping only supports the 'norm' algorithm, got "
        f"{algorithm!r}. 'value'-based clipping is not implemented for "
        f"sharded gradients.")

    fsdp_model = self.trainer.strategy.model
    if not hasattr(fsdp_model, 'clip_grad_norm_'):
      raise RuntimeError(
        "Expected the FSDP-wrapped model to expose clip_grad_norm_ "
        f"(the FSDP-native, shard-aware clipping method); got "
        f"{type(fsdp_model)}. Check the Lightning/PyTorch FSDP version.")

    total_norm = fsdp_model.clip_grad_norm_(float(clip_val))
    self.log(
      name='trainer/grad_norm_fsdp',
      value=float(total_norm),
      on_step=True,
      on_epoch=False,
      sync_dist=False,
      prog_bar=False)

  def _log_training_depth(self):
    global_step = int(getattr(self.trainer, 'global_step', 0) or 0)
    if global_step <= 0 or global_step == self._last_depth_log_step:
      return
    self._last_depth_log_step = global_step

    global_batch_size = int(
      omegaconf.OmegaConf.select(self.config, 'loader.global_batch_size'))
    block_size = int(
      omegaconf.OmegaConf.select(
        self.config, 'block_size', default=self.config.model.length))
    tokens_seen = float(global_step * global_batch_size * block_size)

    self.log('stats/billions_tokens_seen', tokens_seen / 1e9,
             on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)

    max_steps = int(getattr(self.trainer, 'max_steps', 0) or 0)
    if max_steps > 0:
      progress = min(global_step / max_steps, 1.0)
      self.log('stats/progress', progress,
               on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
      if self._train_start_monotonic is not None:
        elapsed_seconds = time.monotonic() - self._train_start_monotonic
        start_progress = min(self._train_start_global_step / max_steps, 1.0)
        completed_progress = progress - start_progress
        self.log('stats/runtime_seconds', elapsed_seconds,
                 on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
        if completed_progress > 0 and progress < 1.0:
          eta_seconds = elapsed_seconds * (1.0 - progress) / completed_progress
        else:
          eta_seconds = 0.0
        self.log('stats/eta_seconds', eta_seconds,
                 on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
        self.log('stats/eta_hours', eta_seconds / 3600.0,
                 on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)

  def on_train_batch_end(self, outputs, batch, batch_idx):
    if not self._first_train_batch_logged:
      self._first_train_batch_logged = True
      if getattr(self.trainer, 'global_rank', 0) == 0:
        elapsed_seconds = None
        if self._train_start_monotonic is not None:
          elapsed_seconds = time.monotonic() - self._train_start_monotonic
        if elapsed_seconds is None:
          logger.info(
            '[TRAIN_TIMING] first training batch finished at %s '
            'batch_idx=%s global_step=%s',
            datetime.now().astimezone().isoformat(timespec='seconds'),
            batch_idx,
            int(getattr(self.trainer, 'global_step', 0) or 0))
        else:
          logger.info(
            '[TRAIN_TIMING] first training batch finished at %s '
            'elapsed_since_train_start_s=%.3f batch_idx=%s global_step=%s',
            datetime.now().astimezone().isoformat(timespec='seconds'),
            elapsed_seconds,
            batch_idx,
            int(getattr(self.trainer, 'global_step', 0) or 0))
    self._log_training_depth()
    self._log_perf_metrics()

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    """Process raw model output into log-probabilities or scores.
    
    Args:
        model_output: Raw output from the backbone model.
        xt: Noisy input tokens.
        sigma: Noise level.
        
    Returns:
        Tensor: Processed output (e.g. log-probs).
    """
    raise NotImplementedError

  def forward(self, xt, sigma, group_idxs=None):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=self._forward_autocast_dtype):
      if group_idxs is None:
        model_output = self.backbone(xt, sigma)
      else:
        model_output = self.backbone(xt, group_idxs, sigma)
    return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    """Generic loss aggregation for all trainer modules."""
    input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
    nll_out = self.nll(input_tokens, current_accumulation_step, train_mode)
    if isinstance(nll_out, tuple):
      aux_metrics = None
      if len(nll_out) == 2:
        loss, elbo = nll_out
      elif len(nll_out) == 3:
        loss, elbo, aux_metrics = nll_out
      else:
        raise ValueError(
          f'Expected nll() to return 2 or 3 values, got {len(nll_out)}.')
    else:
      loss, elbo = nll_out, nll_out
      aux_metrics = None
    assert loss.ndim == 2
    if self.ignore_bos:
      loss[:, 0] = 0
      valid_tokens[:, 0] = 0
      elbo[:, 0] = 0
    if (getattr(self, 'shift_loss_targets', False)
        and valid_tokens.size(-1) == loss.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]
        
    nlls_for_backprop = (loss * valid_tokens).sum()
    nlls = (elbo * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    token_nll = nlls_for_backprop / num_tokens

    return Loss(loss=token_nll,
                nlls=nlls,
                num_tokens=num_tokens,
                aux_metrics=aux_metrics)

  def on_train_epoch_start(self):
    logger.info('[TRAIN_TIMING] on_train_epoch_start '
                '(about to fetch first batch from train_loader)')
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    if self._train_batch_idx_origin is None:
      self._train_batch_idx_origin = int(batch_idx)
    current_accumulation_step = (
      (int(batch_idx) - self._train_batch_idx_origin)
      % self.trainer.accumulate_grad_batches)
    losses = self._loss(batch['input_ids'], batch['attention_mask'], current_accumulation_step, train_mode=True)
    self.metrics.update_train(losses.nlls, losses.num_tokens)
    log_train_aux_metrics = bool(omegaconf.OmegaConf.select(
      self.config, 'training.log_train_aux_metrics', default=True))
    if losses.aux_metrics and log_train_aux_metrics:
      self.log_dict(
        losses.aux_metrics,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        prog_bar=False)
    sync_train_loss = bool(omegaconf.OmegaConf.select(
      self.config, 'training.sync_train_loss', default=True))
    self.log(name='trainer/loss', value=losses.loss, on_step=True,
             on_epoch=False, sync_dist=sync_train_loss, prog_bar=True)
    return losses.loss

  def on_train_epoch_end(self):
    train_metrics = {}
    for k, v in self.metrics.train_nlls.items():
      if getattr(v, 'weight', 0) > 0:
        train_metrics[k] = v.compute()
    if train_metrics:
      self.log_dict(train_metrics, on_step=False, on_epoch=True, sync_dist=True)
    if hasattr(self.metrics, 'train_aux') and self.metrics.train_aux.weight > 0:
      self.log(name='train/aux', value=self.metrics.train_aux.compute(), on_step=False, on_epoch=True, sync_dist=True)

  def on_validation_epoch_start(self):
    self.metrics.reset()
    self._eval_mode()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.num_tokens)
    return losses.loss

  def on_validation_epoch_end(self):
    valid_metrics = {}
    for k, v in self.metrics.valid_nlls.items():
      if getattr(v, 'weight', 0) > 0:
        valid_metrics[k] = v.compute()
    if valid_metrics:
      self.log_dict(valid_metrics, on_step=False, on_epoch=True, sync_dist=True)
    if hasattr(self.metrics, 'valid_aux') and self.metrics.valid_aux.weight > 0:
      self.log(name='val/aux', value=self.metrics.valid_aux.compute(), on_step=False, on_epoch=True, sync_dist=True)
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      try:
        samples, text_samples = None, None
        for _ in range(
          self.config.sampling.num_sample_batches):
          samples = self.generate_samples(num_samples=self.config.loader.eval_batch_size)
          
          self.metrics.record_entropy(samples)
          # For logging and optional saving only
          text_samples = self.tokenizer.batch_decode(samples)
        if text_samples is not None:
          if self.trainer.global_rank == 0 and hasattr(
            self.trainer.logger, 'log_table'):
            # Log the last generated samples
            text_samples = text_samples[
              : self.config.sampling.num_sample_log]
            self.trainer.logger.log_table(
              key=f'samples@global_step{self.global_step}',
              columns=['Generated Samples'],
              data=[[s] for s in text_samples])
          # Always log sample entropy (cheap and useful)
          self.log('val/sample_entropy', self.metrics.sample_entropy.compute(), on_epoch=True, on_step=False, sync_dist=True)

          # Optionally save validation samples for later gen-PPL evaluation
          if getattr(self.config.eval, 'save_validation_samples', False):
            save_dir = Path(os.getcwd()) / 'validation_samples'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'step_{self.global_step}.pt'
            torch.save(samples.detach().cpu(), save_path.as_posix())
      except Exception as e:
        print(f"Sampling failed at step {self.global_step}: {e}")
    self._train_mode()

  def configure_optimizers(self):
    optimizer_name = str(getattr(self.config.optim, 'name', 'adamw')).lower()
    if optimizer_name == 'adamw':
      use_jax_groups = bool(getattr(self.config.optim, "use_jax_param_groups", False))
      adamw_params = (
        self._build_adamw_jax_style_param_groups()
        if use_jax_groups
        else self._get_parameters()
      )
      optimizer = torch.optim.AdamW(
        adamw_params,
        lr=self.config.optim.lr,
        betas=(self.config.optim.beta1,
               self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay)
      
      scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
      scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
      
      
      
    elif optimizer_name == 'scion':
      # ``ScionTrace`` supplies the bounded, periodic rho/sigma collector.
      # Its legacy ``trace`` mode, however, clones every parameter and gradient
      # on *every* optimizer step for a different set of diagnostics.  Keep the
      # two controls separate: a noise probe needs the class but must not turn
      # on the legacy per-step work.
      trace_enabled = bool(getattr(self.config.optim, 'trace_enabled', False))
      collect_noise_stats = bool(getattr(
        self.config.optim, 'trace_collect_noise_stats', False))
      optimizer_cls = ScionTrace if (trace_enabled or collect_noise_stats) else Scion
      optimizer_kwargs = dict(
        lr=self.config.optim.lr,
        momentum=self.config.optim.momentum,
        unconstrained=bool(getattr(self.config.optim, 'unconstrained', False)),
      )
      if optimizer_cls is ScionTrace:
        optimizer_kwargs["trace"] = trace_enabled
        optimizer_kwargs["trace_every"] = int(
          getattr(self.config.optim, 'trace_every', 1))
      optimizer = optimizer_cls(
        self._build_scion_optimizer_groups(),
        **optimizer_kwargs,
      )

      # train_gpt_scion-style LR schedule:
      # 1) linear warmup, 2) constant, 3) linear warmdown, with a minimum LR floor.
      total_steps = int(getattr(self.config.optim, 'num_iterations', self.config.trainer.max_steps))
      if total_steps <= 0:
        raise ValueError(
          f'SCION scheduler requires a positive step budget, got: {total_steps}')
      warmup_iters = int(getattr(self.config.optim, 'warmup_iters', 0))
      warmdown_iters = int(getattr(self.config.optim, 'warmdown_iters', 0))
      min_lr = float(getattr(self.config.optim, 'min_lr', 1e-8))
      min_ratio = (min_lr / self.config.optim.lr)
      if warmdown_iters < 0 or warmup_iters < 0:
        raise ValueError(
          f'SCION scheduler requires a  non negative step warmup and warmdown iterations got: {warmup_iters, warmdown_iters}')

      def _scion_lr_lambda(it):
        it = int(it)
        assert it <= total_steps
        if warmup_iters > 0 and it < warmup_iters:
          ratio = (it + 1) / warmup_iters
        elif warmdown_iters > 0 and it >= (total_steps - warmdown_iters):
          ratio = (total_steps - it) / warmdown_iters
        else:
          ratio = 1.0
        return max(ratio, min_ratio)

      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _scion_lr_lambda)
      scheduler_dict = {'scheduler': scheduler,
                        'interval': 'step',
                        'monitor': 'val/loss',
                        'name': 'trainer/lr'}
      return [optimizer], [scheduler_dict]
    else:
      raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. "
        "Expected one of: adamw, scion.")

    
    return [optimizer], [scheduler_dict]

  def _create_sampler(self):
    """Instantiate (and cache) the configured sampler."""
    if self._sampler is not None:
      return self._sampler
    sampler_cfg = self._sampler_cfg
    if sampler_cfg is None:
      return None
    self._sampler = hydra.utils.instantiate(
      sampler_cfg,
      self.config,
      forward_process=getattr(self, '_forward_process', None),
      _recursive_=False,
    )
    return self._sampler

  def _resolve_sampler_config(self):
    """Return the first sampler config specifying a Hydra target."""
    algo_sampler = getattr(self.config.algo, 'sampler', None)
    if getattr(algo_sampler, '_target_', None):
      return algo_sampler
    sampling_sampler = getattr(self.config.sampling, 'sampler', None)
    if getattr(sampling_sampler, '_target_', None):
      return sampling_sampler
    return None

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=None):
    """Generate samples from the model using the new sampler system.
    
    Subclasses should not need to override this method if they have a 
    corresponding Sampler implementation registered in the sampling registry.
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps
    if eps is None:
      eps = 1e-5
    inject_bos = getattr(self.config.sampling, 'inject_bos', True)
    
    sampler = self._create_sampler()
    if sampler is None:
      raise NotImplementedError(
        f"Algorithm {self.config.algo.name} does not have a configured sampler. "
        "Set 'sampling.sampler._target_' or 'algo.sampler._target_' in the config "
        "to select a Sampler, or override generate_samples().")
    
    return sampler.generate(model=self, num_samples=num_samples, num_steps=num_steps, eps=eps, inject_bos=inject_bos)

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens,
          current_accumulation_step=None, train_mode=False):
    """Compute negative log likelihood for the given input tokens.

    Args:
        input_tokens: Input token indices.
        current_accumulation_step: Current gradient accumulation step index.
        train_mode: Whether the model is in training mode.

    Returns:
        Tensor: NLL loss.
    """
    raise NotImplementedError

class Diffusion(TrainerBase):
  """Base class for diffusion-based algorithms.
  
  Implements continuous-time diffusion logic including time sampling,
  sigma processing, and generic NLL computation.
  """
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.loss_type in {'elbo', 'low_var'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, valid_tokens

  def nll(self, x0,
          current_accumulation_step=None, train_mode=False):
    """Implements diffusion-style NLL evaluation."""
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      t += (1 / self.T)

    alpha_t = self.noise.alpha_t(t)
    dalpha_t = self.noise.alpha_prime_t(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, t)
    # Optional next-token shift: align logits[..., :-1, :] with targets[..., 1:]
    if getattr(self, 'shift_loss_targets', False):
      # MD4-style: compute CE on raw logits, mask to xt==mask, weight by dalpha/(1-alpha).
      # 1) Get raw logits from backbone (bypass post-processing)
      raw_logits = self.backbone(xt, sigma)
      # 2) Apply next-token shift to align logits and targets, also shift xt
      raw_logits, x0, xt = utils.shift_for_next_token(raw_logits, x0, xt)
      # 3) Per-token CE (use log_softmax to avoid adding new imports)
      ce = - raw_logits.log_softmax(-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
      # 4) Mask to only count positions where xt was masked
      mask_positions = (xt == self.mask_id).to(ce.dtype)
      masked_neg_ce = mask_positions * (-ce)
      # 5) Weight by alpha_prime / (1 - alpha)
      weighting = dalpha_t / (1 - alpha_t)
      while weighting.dim() < masked_neg_ce.dim():
        weighting = weighting.unsqueeze(-1)
      return weighting * masked_neg_ce
    else:
      log_x_theta = self.forward(xt, sigma=sigma)
      return self.nll_per_token(
        log_x_theta=log_x_theta,
        xt=xt,
        x0=x0,
        alpha_t=alpha_t,
        dalpha_t=dalpha_t,
        low_var=train_mode and self.loss_type == 'low_var')

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype, device=self.device)
    sigma_t0 = self._sigma_from_alphat(self.noise.alpha_t(t0))
    model_output_t0 = self.forward(x0, sigma_t0)
    return -torch.gather(input=model_output_t0, dim=-1, index=x0[:, :, None]).squeeze(-1)

  def nll_per_token(self, model_output, xt, x0, alpha_t, dalpha_t, low_var):
    """Compute per-token negative log likelihood.

    Args:
        model_output: Model predictions (logits or scores).
        xt: Noisy input tokens.
        x0: Target clean tokens.
        alpha_t: Signal schedule value at time t.
        dalpha_t: Derivative of alpha_t at time t.
        low_var: Whether to use low-variance loss formulation.

    Returns:
        Tensor: Per-token NLL.
    """
    raise NotImplementedError

  def _get_score(self, x, sigma, group_idxs=None):
    raise NotImplementedError


class AbsorbingState(Diffusion):
  """Base class for absorbing state diffusion models (e.g. MDLM).
  
  Handles mask token management and forward process instantiation for
  masking-based methods.
  """
  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do 
    # vocab_size = len(tokenizer), so that we account
    # for the special tokens added in data/loaders.py.
    # But we use tokenizer.vocab_size so as to to be
    # consistent with the prior checkpoints.
    self.mask_id, vocab_size = ensure_mask_token(tokenizer)
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()

    # Instantiate forward process using Hydra
    fp_cfg = getattr(self.config.algo, 'forward_process', None)
    if fp_cfg is None or not hasattr(fp_cfg, '_target_'):
      raise ValueError(
        "Forward process must be configured with '_target_' field. "
        "Example: forward_process._target_=discrete_diffusion.forward_process.AbsorbingForwardProcess"
      )
    fp_config = omegaconf.OmegaConf.create(fp_cfg)
    self._forward_process = hydra.utils.instantiate(
      fp_config,
      tokenizer=self.tokenizer,
      schedule=self.noise,
      _recursive_=False
    )

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    assert not (self.parameterization == 'mean' and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}

  def q_xt(self, x, t):
    """Computes the noisy sample xt by delegating to the configured forward process.
    
    Args:
        x: Clean input tokens [batch, length].
        t: Time values [batch] or float.
        
    Returns:
        Tensor: Noisy tokens xt.
    """
    if not isinstance(t, torch.Tensor):
      t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
    elif t.device != x.device:
      t = t.to(device=x.device)
    out = self._forward_process(x, t)
    xt = out[0] if isinstance(out, (tuple, list)) else out
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    size = batch_dims[0] if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list)) else batch_dims
    return torch.full(tuple(size), self.mask_id, dtype=torch.int64, device=self.device)
