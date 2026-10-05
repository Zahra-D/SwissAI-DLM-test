"""1F1B pipeline-parallel training for the GIDD HF backbone."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import hydra
import omegaconf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
try:
  import wandb
except Exception:
  wandb = None

from ..algorithms import base as trainer_base
from ..algorithms.gidd import (
  GiddLoss,
  GiddLossConstantPi,
  GiddLossEasyDel,
  GiddLossEasyDelLowMem,
  _MaskTokenizerAdapter,
)
from ..data import get_dataloaders, get_tokenizer
from ..models.gidd_hf_wrapper import build_gidd_config
from ..models.modeling_gidd_hf import (
  GiddLayer,
  GiddRMSNorm,
  ScaledLinear,
  compute_basic_frequencies,
)
from ..noise_schedules import HybridDiffusion
from ..noise_schedules.gidd_constant_pi import GiddLinearNoise, sample_t as sample_t_constant_pi
from ..noise_schedules.gidd_easydel import (
  EasyDelHybridDiffusion,
  sample_t as sample_t_easydel,
)
from ..optimizers import Scion, ScionTrace
from .. import utils

logger = utils.get_logger(__name__)


def _trace_rank(rank: int, step: int, message: str):
  logger.info("[rank=%d step=%d] %s", rank, step, message)


def _unused_pipeline_loss(*_args, **_kwargs):
  raise RuntimeError(
    "Internal pipeline error: non-last pipeline stage was asked to compute loss.")


class GiddPipelineStage(nn.Module):
  """A contiguous GIDD transformer slice for one pipeline rank."""

  def __init__(self, config, vocab_size: int, rank: int, world_size: int):
    super().__init__()
    self.hf_config = build_gidd_config(config, vocab_size)
    self.rank = int(rank)
    self.world_size = int(world_size)
    self.is_first = self.rank == 0
    self.is_last = self.rank == self.world_size - 1
    self.hidden_size = int(self.hf_config.hidden_size)
    self.sequence_length = int(self.hf_config.max_position_embeddings)
    self.resid_scale = (
      self.hf_config.resid_scale / self.hf_config.num_hidden_layers)

    start, end = _layer_range(
      num_layers=int(self.hf_config.num_hidden_layers),
      stage_index=self.rank,
      num_stages=self.world_size,
    )
    self.layer_start = start
    self.layer_end = end

    if self.is_first:
      self.embed_tokens = nn.Embedding(
        num_embeddings=self.hf_config.vocab_size,
        embedding_dim=self.hf_config.hidden_size,
      )
      self.embed_tokens.weight.data = self.embed_tokens.weight.data.to(
        self.hf_config.torch_dtype)
      nn.init.normal_(
        self.embed_tokens.weight,
        mean=0.0,
        std=self.hf_config.emb_init_scale,
      )
    else:
      self.embed_tokens = None

    self.layers = nn.ModuleList([
      GiddLayer(
        config=self.hf_config,
        layer_idx=i,
        resid_scale=self.resid_scale,
        dtype=self.hf_config.torch_dtype,
      )
      for i in range(start, end)
    ])

    freqs = compute_basic_frequencies(
      base=self.hf_config.rope_theta,
      rotary_dim=(
        self.hf_config.hidden_size // self.hf_config.num_attention_heads),
      max_position_embeddings=self.hf_config.max_position_embeddings,
    )
    self.frequencies = nn.Buffer(freqs, persistent=False)

    if self.is_last:
      self.norm = GiddRMSNorm(config=self.hf_config, dtype=torch.float32)
      self.lm_head = ScaledLinear(
        self.hf_config.hidden_size,
        self.hf_config.vocab_size,
        scale=self.hf_config.head_scaling,
        dtype=self.hf_config.torch_dtype,
        use_bias=False,
        init_std=self.hf_config.head_init_scale,
      )
    else:
      self.norm = None
      self.lm_head = None

    self.activation_checkpointing = bool(
      self.hf_config.activation_checkpointing)
    self.activation_checkpoint_preserve_rng_state = bool(
      self.hf_config.activation_checkpoint_preserve_rng_state)

  def _checkpoint_layer(
      self,
      block: GiddLayer,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor) -> torch.Tensor:

    def custom_forward(hidden_states: torch.Tensor) -> torch.Tensor:
      return block(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        output_attentions=False,
        frequencies=self.frequencies,
        past_key_values=None,
      ).hidden_states

    return torch_checkpoint.checkpoint(
      custom_forward,
      hidden_states,
      use_reentrant=False,
      preserve_rng_state=self.activation_checkpoint_preserve_rng_state,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.is_first:
      hidden_states = self.embed_tokens(x.to(torch.long))
    else:
      hidden_states = x

    if hidden_states.shape[1] > self.hf_config.max_position_embeddings:
      raise ValueError(
        "Maximum position embedding reached: "
        f"{hidden_states.shape[1]} > {self.hf_config.max_position_embeddings}")

    position_ids = torch.arange(
      hidden_states.shape[1],
      device=hidden_states.device,
      dtype=torch.long,
    ).unsqueeze(0)

    use_checkpoint = self.activation_checkpointing and self.training
    for block in self.layers:
      if use_checkpoint:
        hidden_states = self._checkpoint_layer(
          block=block,
          hidden_states=hidden_states,
          position_ids=position_ids,
        )
      else:
        hidden_states = block(
          hidden_states=hidden_states,
          attention_mask=None,
          position_ids=position_ids,
          output_attentions=False,
          frequencies=self.frequencies,
          past_key_values=None,
        ).hidden_states

    if not self.is_last:
      return hidden_states

    hidden_states = self.norm(hidden_states)
    if self.hf_config.tie_word_embeddings:
      if self.embed_tokens is None:
        raise ValueError(
          "Pipeline mode does not support tie_word_embeddings=True because "
          "the embedding weights live on the first stage.")
      return hidden_states @ self.embed_tokens.weight.t()
    return self.lm_head(hidden_states)


def _layer_range(num_layers: int, stage_index: int, num_stages: int) -> tuple[int, int]:
  per_stage = num_layers // num_stages
  remainder = num_layers % num_stages
  start = stage_index * per_stage + min(stage_index, remainder)
  end = start + per_stage + (1 if stage_index < remainder else 0)
  return start, end


def _is_gidd_hf_model(config) -> bool:
  model_name = str(getattr(config.model, 'name', '')).lower()
  model_type = str(getattr(config.model, 'type', '')).lower()
  model_target = str(getattr(config.model, '_target_', '')).lower()
  return (
    'gidd_hf' in model_name
    or 'gidd_hf' in model_type
    or 'gidd_hf_wrapper' in model_target
  )


def _require_pipeline_api():
  try:
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B
  except Exception as exc:
    raise RuntimeError(
      "parallel.pipeline.enabled=true requires PyTorch's "
      "torch.distributed.pipelining API. This repo pins torch==2.7.0; "
      "make sure the training environment installed that build.") from exc
  return PipelineStage, Schedule1F1B


def _init_distributed():
  if not torch.cuda.is_available():
    raise RuntimeError("1F1B pipeline training currently requires CUDA.")
  if not dist.is_available():
    raise RuntimeError("torch.distributed is not available in this PyTorch build.")
  if not dist.is_initialized():
    rank = int(os.environ.get(
      "RANK",
      os.environ.get("SLURM_PROCID", "0"),
    ))
    world_size = int(os.environ.get(
      "WORLD_SIZE",
      os.environ.get("SLURM_NTASKS", "1"),
    ))
    local_rank = int(os.environ.get(
      "LOCAL_RANK",
      os.environ.get("SLURM_LOCALID", str(rank % max(1, torch.cuda.device_count()))),
    ))

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    if world_size > 1:
      os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
      os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
      backend=os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"),
      rank=rank,
      world_size=world_size,
      device_id=torch.device("cuda", local_rank),
    )
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  local_rank = int(os.environ.get(
    "LOCAL_RANK",
    os.environ.get("SLURM_LOCALID", str(rank % max(1, torch.cuda.device_count()))),
  ))
  torch.cuda.set_device(local_rank)
  return rank, world_size, torch.device("cuda", local_rank)


def _pipeline_cfg(config):
  return omegaconf.OmegaConf.select(config, 'parallel.pipeline', default={})


def _num_microbatches(config) -> int:
  value = int(omegaconf.OmegaConf.select(
    config, 'parallel.pipeline.num_microbatches', default=0) or 0)
  if value <= 0:
    raise ValueError("parallel.pipeline.num_microbatches must be positive.")
  return value


def _batch_size_per_pipeline_step(config) -> int:
  explicit = omegaconf.OmegaConf.select(
    config, 'parallel.pipeline.batch_size', default=None)
  if explicit is not None:
    return int(explicit)
  accum = max(1, int(getattr(config.trainer, 'accumulate_grad_batches', 1) or 1))
  global_batch_size = int(config.loader.global_batch_size)
  if global_batch_size % accum != 0:
    raise ValueError(
      "loader.global_batch_size must be divisible by "
      "trainer.accumulate_grad_batches for pipeline training.")
  return global_batch_size // accum


def _make_loader_config(config, batch_size: int):
  loader_config = omegaconf.OmegaConf.create(omegaconf.OmegaConf.to_container(
    config, resolve=False))
  omegaconf.OmegaConf.set_struct(loader_config, False)
  visible_devices = max(1, torch.cuda.device_count())
  loader_config.loader.batch_size = batch_size
  loader_config.loader.eval_batch_size = batch_size
  # get_dataloaders validates Lightning's DDP formula. Pipeline mode only
  # constructs a real DataLoader on rank 0, so these global values are shimmed
  # to satisfy that validation while preserving the actual DataLoader batch.
  loader_config.loader.global_batch_size = batch_size * visible_devices
  loader_config.loader.eval_global_batch_size = batch_size * visible_devices
  loader_config.trainer.devices = 1
  loader_config.trainer.num_nodes = 1
  loader_config.trainer.accumulate_grad_batches = 1
  return loader_config


def _build_noise_and_loss(config, tokenizer, vocab_size: int, *, need_loss: bool):
  mask_tok = _MaskTokenizerAdapter(
    base_tokenizer=tokenizer,
    vocab_size=vocab_size,
    mask_token_id=tokenizer.mask_token_id,
  )
  p_uniform = float(getattr(config.algo, 'p_uniform', None))
  loss_type = str(getattr(config.algo, 'loss_type', None))

  if loss_type == 'gidd_constant_pi':
    noise = GiddLinearNoise(tokenizer=mask_tok, p_uniform=p_uniform)
    sample_t = sample_t_constant_pi
    loss_fn = GiddLossConstantPi(config, mask_tok, noise) if need_loss else None
  elif loss_type in {'gidd_easydel', 'gidd_easydel_lowmem'}:
    noise = EasyDelHybridDiffusion(
      tokenizer=mask_tok,
      min_log_snr=float(getattr(config.algo, 'min_log_snr', -10.0)),
      max_log_snr=float(getattr(config.algo, 'max_log_snr', 10.0)),
      hybrid_scale=float(getattr(config.algo, 'hybrid_mixing_scale', 1.0)),
      hybrid_shift=float(getattr(config.algo, 'hybrid_mixing_shift', 0.0)),
      prior_distribution=str(getattr(config.algo, 'prior_distribution', 'masked')),
    )
    sample_t = sample_t_easydel
    if need_loss:
      loss_cls = GiddLossEasyDelLowMem if loss_type == 'gidd_easydel_lowmem' else GiddLossEasyDel
      loss_fn = loss_cls(config, mask_tok, noise)
    else:
      loss_fn = None
  elif loss_type == 'gidd':
    noise = HybridDiffusion(
      tokenizer=mask_tok,
      p_uniform=p_uniform,
      clip_noise=20,
      gamma=1.0,
    )
    from ..noise_schedules import sample_t as sample_t_hybrid
    sample_t = sample_t_hybrid
    loss_fn = GiddLoss(config, mask_tok, noise) if need_loss else None
  else:
    raise ValueError(
      f"Pipeline GIDD does not understand algo.loss_type={loss_type!r}.")

  return noise, loss_fn, sample_t


def _build_optimizer(config, parameters: Iterable[torch.nn.Parameter], stage: nn.Module):
  optimizer_name = str(getattr(config.optim, 'name', 'adamw')).lower()
  params = list(parameters)
  if optimizer_name == 'adamw':
    optimizer = torch.optim.AdamW(
      params,
      lr=config.optim.lr,
      betas=(config.optim.beta1, config.optim.beta2),
      eps=config.optim.eps,
      weight_decay=config.optim.weight_decay,
    )
    scheduler = hydra.utils.instantiate(config.lr_scheduler, optimizer=optimizer)
    return optimizer, scheduler

  if optimizer_name == 'scion':
    trace_enabled = bool(getattr(config.optim, 'trace_enabled', False))
    optimizer_cls = ScionTrace if trace_enabled else Scion
    optimizer = optimizer_cls(
      _build_pipeline_scion_groups(config, stage),
      lr=config.optim.lr,
      momentum=config.optim.momentum,
      unconstrained=bool(getattr(config.optim, 'unconstrained', False)),
      **({"trace": True} if trace_enabled else {}),
    )
    if bool(getattr(config.optim, 'boundary_init', False)):
      # Pipeline stages are complete tensors and are already resident on their
      # assigned GPU here, so Section B.3 initialization is both correct and
      # substantially faster than performing the QR draws on CPU.
      optimizer.init()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      _scion_lr_lambda(config),
    )
    return optimizer, scheduler

  raise ValueError(
    f"Unsupported optimizer '{optimizer_name}' for pipeline training.")


def _build_pipeline_scion_groups(config, stage: nn.Module):
  optim_cfg = config.optim
  hidden_size = config.model.hidden_size
  if bool(getattr(optim_cfg, 'equal_group_lr', False)):
    matrix_lr = optim_cfg.lr
    aux_lr = optim_cfg.lr
  else:
    matrix_lr = optim_cfg.lr / hidden_size
    aux_lr = optim_cfg.lr * optim_cfg.aux_lr_factor

  grouped = {
    'embedding': [],
    'bias': [],
    'one_d': [],
    'matrix': [],
  }
  for name, param in stage.named_parameters():
    if not param.requires_grad:
      continue
    lname = name.lower()
    if 'embed_tokens' in lname or 'lm_head' in lname:
      grouped['embedding'].append(param)
    elif 'bias' in lname:
      grouped['bias'].append(param)
    elif param.ndim <= 1:
      grouped['one_d'].append(param)
    else:
      grouped['matrix'].append(param)

  groups = []
  if grouped['embedding']:
    groups.append({
      'params': grouped['embedding'],
      'name': 'embedding',
      'norm': 'Sign',
      'norm_kwargs': {},
      'scale': optim_cfg.scale_embed,
      'lr': aux_lr,
    })
  if grouped['bias']:
    groups.append({
      'params': grouped['bias'],
      'name': 'bias',
      'norm': optim_cfg.bias_norm,
      'norm_kwargs': {},
      'scale': optim_cfg.scale_bias,
      'lr': aux_lr,
    })
  if grouped['one_d']:
    groups.append({
      'params': grouped['one_d'],
      'name': 'one_d',
      'norm': optim_cfg.norm_layer_norm,
      'norm_kwargs': {},
      'scale': optim_cfg.scale_layer_norm,
      'lr': aux_lr,
    })
  if grouped['matrix']:
    groups.append({
      'params': grouped['matrix'],
      'name': 'matrix',
      'norm': 'Spectral',
      'norm_kwargs': {'steps': optim_cfg.spectral_norm_steps},
      'scale': optim_cfg.scale_matrix,
      'lr': matrix_lr,
    })
  if not groups:
    raise ValueError("Pipeline stage has no trainable SCION parameters.")
  return groups


def _scion_lr_lambda(config):
  total_steps = int(getattr(config.optim, 'num_iterations', config.trainer.max_steps))
  warmup_iters = int(getattr(config.optim, 'warmup_iters', 0))
  warmdown_iters = int(getattr(config.optim, 'warmdown_iters', 0))
  min_lr = float(getattr(config.optim, 'min_lr', 1e-8))
  min_ratio = min_lr / float(config.optim.lr)

  def lr_lambda(step):
    step = int(step)
    if warmup_iters > 0 and step < warmup_iters:
      ratio = (step + 1) / warmup_iters
    elif warmdown_iters > 0 and step >= (total_steps - warmdown_iters):
      ratio = (total_steps - step) / warmdown_iters
    else:
      ratio = 1.0
    return max(ratio, min_ratio)

  return lr_lambda


def _sample_and_broadcast_batch(
    *,
    rank: int,
    device: torch.device,
    batch_iter,
    loader,
    batch_size: int,
    seq_len: int,
    sample_t,
    noise,
    config,
):
  input_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
  attention_mask = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
  z_t = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
  t = torch.empty((batch_size,), dtype=torch.float32, device=device)

  if rank == 0:
    try:
      batch = next(batch_iter)
    except StopIteration:
      batch_iter = iter(loader)
      batch = next(batch_iter)
    ids = batch['input_ids'].to(device=device, non_blocking=True)
    mask = batch['attention_mask'].to(device=device, non_blocking=True)
    if ids.shape != input_ids.shape:
      raise ValueError(
        "Pipeline training requires static batch shapes. "
        f"Expected {tuple(input_ids.shape)}, got {tuple(ids.shape)}. "
        "Set data.drop_last or choose a dataset/batch size without a short final batch.")
    input_ids.copy_(ids)
    attention_mask.copy_(mask)
    t.copy_(sample_t(config, batch_size, eps=float(getattr(config.algo, 't_eps', 1e-4)), device=device))
    z_t.copy_(noise.sample_zt(input_ids, t))

  for tensor in (input_ids, attention_mask, z_t, t):
    dist.broadcast(tensor, src=0)

  return input_ids, attention_mask, z_t, t, batch_iter


def _make_loss_closure(loss_module, accumulation_steps: int):
  target_queue = []
  metric_queue = []

  def enqueue_targets(
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      z_t: torch.Tensor,
      t: torch.Tensor,
      *,
      n_microbatches: int):
    target_queue.clear()
    metric_queue.clear()
    input_ids_split = torch.tensor_split(input_ids, n_microbatches, dim=0)
    attention_mask_split = torch.tensor_split(attention_mask, n_microbatches, dim=0)
    z_t_split = torch.tensor_split(z_t, n_microbatches, dim=0)
    t_split = torch.tensor_split(t, n_microbatches, dim=0)
    target_queue.extend(zip(input_ids_split, attention_mask_split, z_t_split, t_split))

  def loss_fn(logits: torch.Tensor, _target):
    if not target_queue:
      raise RuntimeError(
        "Pipeline loss target queue is empty. "
        "Expected per-microbatch targets to be enqueued before schedule.step().")
    input_ids, attention_mask, z_t, t = target_queue.pop(0)
    loss, _, metrics = loss_module(
      logits=logits,
      input_ids=input_ids,
      attention_mask=attention_mask.to(logits.dtype),
      z_t=z_t,
      t=t,
      reduction='tokenmean',
    )
    if metrics:
      metric_queue.append({
        key: float(value.detach().float().item())
        for key, value in metrics.items()
      })
    return loss / accumulation_steps

  def pop_metrics():
    values = list(metric_queue)
    metric_queue.clear()
    return values

  return loss_fn, enqueue_targets, pop_metrics


def _gather_to_rank0(rank: int, world_size: int, payload):
  gathered = [None] * world_size if rank == 0 else None
  dist.gather_object(payload, gathered, dst=0)
  return gathered


def _pipeline_effective_global_batch_size(batch_size: int, accumulation_steps: int) -> int:
  return int(batch_size) * int(accumulation_steps)


def _estimate_pipeline_flops_per_token(config, trainable_params: int, embedding_params: int) -> float:
  manual_flops = omegaconf.OmegaConf.select(
    config, 'perf.flops_per_token', default=None)
  if manual_flops is not None:
    return float(manual_flops)

  dense_trainable_params = max(0, int(trainable_params) - int(embedding_params))
  model_cfg = config.model
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
      config, 'block_size', default=config.model.length))

  parameter_flops = 6.0 * float(dense_trainable_params)
  attention_flops = 0.0
  if layers > 0 and num_heads > 0 and head_dim > 0 and block_size > 0:
    attention_flops = 12.0 * layers * num_heads * head_dim * block_size
  return parameter_flops + attention_flops


def _collect_stage_param_stats(stage_module: nn.Module):
  model_params = 0
  trainable_params = 0
  embedding_params = 0
  for module in stage_module.modules():
    if isinstance(module, nn.Embedding):
      embedding_params += sum(param.numel() for param in module.parameters(recurse=False))
  for param in stage_module.parameters():
    model_params += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  return {
    'model_params': int(model_params),
    'trainable_params': int(trainable_params),
    'embedding_params': int(embedding_params),
  }


def _log_perf_static_metadata(
    config,
    *,
    rank: int,
    world_size: int,
    wandb_run,
    stage_module: nn.Module,
    batch_size: int,
    accumulation_steps: int,
    seq_len: int):
  if not bool(omegaconf.OmegaConf.select(config, 'perf.enabled', default=False)):
    return

  gathered = _gather_to_rank0(rank, world_size, _collect_stage_param_stats(stage_module))
  if rank != 0 or wandb_run is None:
    return

  total_model_params = sum(int(item['model_params']) for item in gathered)
  total_trainable_params = sum(int(item['trainable_params']) for item in gathered)
  total_embedding_params = sum(int(item['embedding_params']) for item in gathered)
  flops_per_token = _estimate_pipeline_flops_per_token(
    config,
    trainable_params=total_trainable_params,
    embedding_params=total_embedding_params,
  )
  effective_global_batch_size = _pipeline_effective_global_batch_size(
    batch_size, accumulation_steps)
  tokens_per_step = float(effective_global_batch_size * seq_len)
  metrics = {
    'perf_constants/model_params': float(total_model_params),
    'perf_constants/trainable_params': float(total_trainable_params),
    'perf_constants/dense_trainable_params_estimate': float(
      max(0, total_trainable_params - total_embedding_params)),
    'perf_constants/flops_per_token_estimate': float(flops_per_token),
    'perf_constants/flops_per_step_estimate': float(flops_per_token) * tokens_per_step,
    'perf_constants/world_size': float(world_size),
    'perf_constants/gpu_count': float(world_size),
    'perf_constants/global_batch_size': float(effective_global_batch_size),
    'perf_constants/micro_batch_size_per_gpu': float(batch_size // max(1, _num_microbatches(config))),
    'perf_constants/grad_accum_steps': float(accumulation_steps),
    'perf_constants/seq_len': float(seq_len),
  }
  peak_tflops = omegaconf.OmegaConf.select(
    config, 'perf.theoretical_peak_tflops_per_gpu', default=None)
  if peak_tflops is not None:
    metrics['perf_constants/theoretical_peak_tflops_per_gpu'] = float(peak_tflops)
  wandb_run.log(metrics, step=0)
  # return metrics


def _collect_optimizer_logs(optimizer):
  metrics = {}
  for group_idx, group in enumerate(getattr(optimizer, 'param_groups', [])):
    group_name = str(group.get('name', f'group_{group_idx}'))
    metrics[f'trainer/lr/{group_name}'] = float(group.get('lr', 0.0))

  if hasattr(optimizer, 'pop_trace_logs'):
    for key, value in optimizer.pop_trace_logs().items():
      metrics[str(key)] = float(value)
  if hasattr(optimizer, 'pop_noise_logs'):
    for key, value in optimizer.pop_noise_logs().items():
      metrics[str(key)] = float(value)
  return metrics


def _summarize_metric_list(metric_list):
  if not metric_list:
    return {}
  totals = {}
  counts = {}
  for metrics in metric_list:
    for key, value in metrics.items():
      totals[key] = totals.get(key, 0.0) + float(value)
      counts[key] = counts.get(key, 0) + 1
  return {key: totals[key] / max(1, counts[key]) for key in totals}


def _collect_rank_perf_stats(device: torch.device):
  if not (torch.cuda.is_available() and getattr(device, 'type', None) == 'cuda'):
    return {}
  return {
    'cuda_memory_allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
    'cuda_memory_reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
    'max_cuda_memory_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
    'max_cuda_memory_reserved_gb': torch.cuda.max_memory_reserved(device) / 1e9,
  }


def _collect_step_diagnostics(
    *,
    optimizer,
    device: torch.device,
    need_loss: bool,
    step_aux_metrics):
  return {
    'aux': _summarize_metric_list(step_aux_metrics) if need_loss else {},
    'lrs': _collect_optimizer_logs(optimizer),
    'mem': _collect_rank_perf_stats(device),
  }


def _build_progress_metrics(
    *,
    global_step: int,
    max_steps: int,
    train_start_step: int,
    train_start_time: float | None,
    effective_global_batch_size: int,
    seq_len: int):
  metrics = {
    'stats/billions_tokens_seen': (
      float(global_step * effective_global_batch_size * seq_len) / 1e9),
  }
  if max_steps > 0:
    progress = min(float(global_step) / float(max_steps), 1.0)
    metrics['stats/progress'] = progress
    if train_start_time is not None:
      elapsed_seconds = time.monotonic() - train_start_time
      metrics['stats/runtime_seconds'] = elapsed_seconds
      start_progress = min(float(train_start_step) / float(max_steps), 1.0)
      completed_progress = progress - start_progress
      if completed_progress > 0 and progress < 1.0:
        eta_seconds = elapsed_seconds * (1.0 - progress) / completed_progress
      else:
        eta_seconds = 0.0
      metrics['stats/eta_seconds'] = eta_seconds
      metrics['stats/eta_hours'] = eta_seconds / 3600.0
  return metrics


def _stage_example_args(
    *,
    rank: int,
    world_size: int,
    microbatch_size: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
  if rank == 0:
    input_args = (torch.zeros(
      microbatch_size, seq_len, dtype=torch.long, device=device),)
  else:
    input_args = (torch.zeros(
      microbatch_size, seq_len, hidden_size, dtype=dtype, device=device),)

  if rank == world_size - 1:
    output_args = torch.zeros(
      microbatch_size, seq_len, vocab_size, dtype=dtype, device=device)
  else:
    output_args = torch.zeros(
      microbatch_size, seq_len, hidden_size, dtype=dtype, device=device)

  return input_args, output_args


def _checkpoint_path(config, rank: int) -> Path:
  return Path(config.checkpointing.save_dir) / "pipeline_1f1b" / f"rank{rank:04d}.pt"


def _maybe_load_checkpoint(config, rank: int, stage, optimizer, scheduler):
  if not bool(getattr(config.checkpointing, 'resume_from_ckpt', False)):
    return 0
  path = _checkpoint_path(config, rank)
  if not path.exists():
    return 0
  checkpoint = torch.load(path, map_location='cpu')
  stage.load_state_dict(checkpoint['stage'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  if checkpoint.get('scheduler') is not None:
    scheduler.load_state_dict(checkpoint['scheduler'])
  return int(checkpoint.get('global_step', 0))


def _save_checkpoint(config, rank: int, stage, optimizer, scheduler, global_step: int):
  path = _checkpoint_path(config, rank)
  if rank == 0:
    path.parent.mkdir(parents=True, exist_ok=True)
  dist.barrier()
  torch.save({
    'global_step': int(global_step),
    'stage': stage.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': None if scheduler is None else scheduler.state_dict(),
    'rank': rank,
    'world_size': dist.get_world_size(),
    'config': omegaconf.OmegaConf.to_container(config, resolve=True),
  }, path)
  dist.barrier()


def _init_wandb(config, rank: int):
  if rank != 0 or config.get('wandb', None) is None:
    return None
  if wandb is None:
    logger.warning("wandb is not installed/importable; skipping W&B logging.")
    return None

  wandb_config = omegaconf.OmegaConf.to_container(config, resolve=True)
  settings = omegaconf.OmegaConf.to_container(config.wandb, resolve=True) or {}
  save_dir = settings.pop('save_dir', None)
  settings = {key: value for key, value in settings.items() if value is not None}
  if save_dir is not None:
    os.makedirs(str(save_dir), exist_ok=True)
  run = wandb.init(
    config=wandb_config,
    dir=save_dir,
    **settings,
  )
  return run


def train_pipeline_1f1b(config):
  """Train the GIDD HF model using one pipeline stage per distributed rank."""
  PipelineStage, Schedule1F1B = _require_pipeline_api()
  if not _is_gidd_hf_model(config):
    raise ValueError(
      "parallel.pipeline.enabled=true currently supports the GiddHFWrapper "
      "model family only.")
  if bool(getattr(config.model, 'tie_word_embeddings', False)):
    raise ValueError(
      "Pipeline mode currently requires model.tie_word_embeddings=false.")

  rank, world_size, device = _init_distributed()
  if world_size < 2:
    raise ValueError("1F1B pipeline training requires at least 2 ranks.")

  torch.set_float32_matmul_precision("high")
  tokenizer = get_tokenizer(config)
  _, vocab_size = trainer_base.ensure_mask_token(tokenizer)

  batch_size = _batch_size_per_pipeline_step(config)
  n_microbatches = _num_microbatches(config)
  accumulation_steps = max(
    1, int(getattr(config.trainer, 'accumulate_grad_batches', 1) or 1))
  if batch_size % n_microbatches != 0:
    raise ValueError(
      "Pipeline batch size must be divisible by "
      f"parallel.pipeline.num_microbatches ({batch_size} vs {n_microbatches}).")
  microbatch_size = batch_size // n_microbatches
  seq_len = int(config.model.length)

  train_loader = None
  train_iter = None
  if rank == 0:
    loader_config = _make_loader_config(config, batch_size)
    train_loader, _ = get_dataloaders(loader_config, tokenizer, skip_valid=True)
    train_iter = iter(train_loader)

  need_loss = rank == world_size - 1
  need_noise = rank == 0 or need_loss
  noise, loss_module, sample_t = (
    _build_noise_and_loss(config, tokenizer, vocab_size, need_loss=need_loss)
    if need_noise
    else (None, None, None)
  )
  if noise is not None:
    noise = noise.to(device)
  if loss_module is not None:
    loss_module = loss_module.to(device)

  stage_module = GiddPipelineStage(config, vocab_size, rank, world_size).to(device)
  stage_module.train()
  optimizer, scheduler = _build_optimizer(
    config,
    (p for p in stage_module.parameters() if p.requires_grad),
    stage_module,
  )

  input_args, output_args = _stage_example_args(
    rank=rank,
    world_size=world_size,
    microbatch_size=microbatch_size,
    seq_len=seq_len,
    hidden_size=int(config.model.hidden_size),
    vocab_size=vocab_size,
    dtype=stage_module.hf_config.torch_dtype,
    device=device,
  )
  stage = PipelineStage(
    stage_module,
    stage_index=rank,
    num_stages=world_size,
    device=device,
    input_args=input_args,
    output_args=output_args,
  )
  loss_fn = None
  enqueue_targets = None
  pop_loss_metrics = None
  if need_loss:
    loss_fn, enqueue_targets, pop_loss_metrics = _make_loss_closure(
      loss_module, accumulation_steps)

  schedule = Schedule1F1B(
    stage,
    n_microbatches=n_microbatches,
    # Schedule1F1B uses loss_fn presence to configure backward-capable
    # execution. Only the final stage should actually invoke it.
    loss_fn=loss_fn if need_loss else _unused_pipeline_loss,
    scale_grads=True,
  )

  global_step = _maybe_load_checkpoint(
    config, rank, stage_module, optimizer, scheduler)
  max_steps = int(getattr(config.trainer, 'max_steps', 0) or 0)
  if max_steps <= 0:
    raise ValueError("trainer.max_steps must be positive for pipeline training.")
  log_every = max(1, int(getattr(config.trainer, 'log_every_n_steps', 100) or 100))
  checkpoint_every = int(omegaconf.OmegaConf.select(
    config, 'parallel.pipeline.checkpoint_every_n_steps', default=0) or 0)

  if rank == 0:
    logger.info(
      "Starting 1F1B pipeline training: stages=%d batch=%d microbatches=%d "
      "accumulation=%d",
      world_size,
      batch_size,
      n_microbatches,
      accumulation_steps,
    )
  wandb_run = _init_wandb(config, rank)
  if rank == 0 and wandb_run is not None:
    wandb_run.log({
      'system/pipeline_initialized': 1.0,
      'system/world_size': float(world_size),
      'system/pipeline_batch_size': float(batch_size),
      'system/pipeline_microbatch_size': float(microbatch_size),
      'system/num_microbatches': float(n_microbatches),
    }, step=global_step)
  _log_perf_static_metadata(
    config,
    rank=rank,
    world_size=world_size,
    wandb_run=wandb_run,
    stage_module=stage_module,
    batch_size=batch_size,
    accumulation_steps=accumulation_steps,
    seq_len=seq_len,
  )

  dist.barrier()
  step_start = time.perf_counter()
  train_start_step = global_step
  train_start_time = time.monotonic()
  effective_global_batch_size = _pipeline_effective_global_batch_size(
    batch_size, accumulation_steps)
  flops_per_token = None
  if bool(omegaconf.OmegaConf.select(config, 'perf.enabled', default=False)):
    stage_stats = _collect_stage_param_stats(stage_module)
    gathered_stats = _gather_to_rank0(rank, world_size, stage_stats)
    if rank == 0:
      flops_per_token = _estimate_pipeline_flops_per_token(
        config,
        trainable_params=sum(int(item['trainable_params']) for item in gathered_stats),
        embedding_params=sum(int(item['embedding_params']) for item in gathered_stats),
      )
    flops_box = [flops_per_token] if rank == 0 else [0.0]
    dist.broadcast_object_list(flops_box, src=0)
    flops_per_token = float(flops_box[0])

  while global_step < max_steps:
    optimizer.zero_grad(set_to_none=True)
    step_losses = []
    step_aux_metrics = []
    for accum_idx in range(accumulation_steps):
      trace_this_iter = global_step == 0 and accum_idx == 0
      if trace_this_iter:
        _trace_rank(rank, global_step, "enter batch broadcast")
      input_ids, attention_mask, z_t, t, train_iter = _sample_and_broadcast_batch(
        rank=rank,
        device=device,
        batch_iter=train_iter,
        loader=train_loader,
        batch_size=batch_size,
        seq_len=seq_len,
        sample_t=sample_t,
        noise=noise,
        config=config,
      )
      if trace_this_iter:
        _trace_rank(
          rank,
          global_step,
          f"broadcast done input_ids={tuple(input_ids.shape)} z_t={tuple(z_t.shape)} t={tuple(t.shape)}",
        )
      if rank == 0:
        if trace_this_iter:
          _trace_rank(rank, global_step, "before schedule.step rank0 input")
        schedule.step(z_t)
        if trace_this_iter:
          _trace_rank(rank, global_step, "after schedule.step rank0 input")
      elif need_loss:
        losses = []
        enqueue_targets(
          input_ids=input_ids,
          attention_mask=attention_mask,
          z_t=z_t,
          t=t,
          n_microbatches=n_microbatches,
        )
        if trace_this_iter:
          _trace_rank(rank, global_step, "before schedule.step last-rank target/loss")
        schedule.step(
          target=input_ids,
          losses=losses,
        )
        if trace_this_iter:
          _trace_rank(rank, global_step, f"after schedule.step last-rank losses={len(losses)}")
        if losses:
          step_losses.extend([loss.detach() for loss in losses])
        if pop_loss_metrics is not None:
          step_aux_metrics.extend(pop_loss_metrics())
      else:
        if trace_this_iter:
          _trace_rank(rank, global_step, "before schedule.step middle-rank")
        schedule.step()
        if trace_this_iter:
          _trace_rank(rank, global_step, "after schedule.step middle-rank")

    if float(getattr(config.trainer, 'gradient_clip_val', 0.0) or 0.0) > 0:
      torch.nn.utils.clip_grad_norm_(
        stage_module.parameters(),
        float(config.trainer.gradient_clip_val),
      )
    optimizer.step()
    scheduler.step()
    global_step += 1

    if global_step % log_every == 0 or global_step == 1:
      elapsed = time.perf_counter() - step_start
      step_start = time.perf_counter()
      loss_value = torch.zeros((), dtype=torch.float32, device=device)
      if need_loss and step_losses:
        loss_value = torch.stack([x.float() for x in step_losses]).mean()
      dist.broadcast(loss_value, src=world_size - 1)
      gathered_diagnostics = _gather_to_rank0(
        rank,
        world_size,
        _collect_step_diagnostics(
          optimizer=optimizer,
          device=device,
          need_loss=need_loss,
          step_aux_metrics=step_aux_metrics,
        ),
      )
      if rank == 0:
        tokens_per_step = effective_global_batch_size * seq_len
        tokens_per_second = tokens_per_step / max(elapsed, 1e-12)
        metric_payload = {
          "trainer/loss": float(loss_value.item()),
          "train/loss": float(loss_value.item()),
          "train/step_time": elapsed,
          "train/tokens_per_second": tokens_per_second,
          "train/tokens_per_second_per_gpu": tokens_per_second / world_size,
          "train/pipeline_batch_size": batch_size,
          "train/pipeline_microbatch_size": microbatch_size,
          "train/num_microbatches": n_microbatches,
          "train/pipeline_stages": world_size,
        }
        metric_payload.update(_build_progress_metrics(
          global_step=global_step,
          max_steps=max_steps,
          train_start_step=train_start_step,
          train_start_time=train_start_time,
          effective_global_batch_size=effective_global_batch_size,
          seq_len=seq_len,
        ))
        for stage_idx, stage_payload in enumerate(gathered_diagnostics or []):
          if not stage_payload:
            continue
          stage_metrics = stage_payload.get('aux', {})
          for key, value in stage_metrics.items():
            metric_payload[key] = float(value)
            metric_payload[f"{key}/stage_{stage_idx}"] = float(value)
        for stage_idx, stage_payload in enumerate(gathered_diagnostics or []):
          if not stage_payload:
            continue
          stage_metrics = stage_payload.get('lrs', {})
          for key, value in stage_metrics.items():
            metric_payload[f"{key}/stage_{stage_idx}"] = float(value)
        if bool(omegaconf.OmegaConf.select(config, 'perf.enabled', default=False)):
          metric_payload.update({
            'perf/step_time_s': elapsed,
            'perf/tokens_per_s': tokens_per_second,
            'perf/tokens_per_s_per_gpu': tokens_per_second / max(world_size, 1),
            'perf/samples_per_s': float(effective_global_batch_size) / max(elapsed, 1e-12),
          })
          if flops_per_token is not None:
            flops_per_step = float(flops_per_token) * float(tokens_per_step)
            achieved_flops_per_s = flops_per_step / max(elapsed, 1e-12)
            achieved_flops_per_s_per_gpu = achieved_flops_per_s / max(world_size, 1)
            metric_payload['perf/achieved_tflops'] = achieved_flops_per_s / 1e12
            metric_payload['perf/achieved_tflops_per_gpu'] = (
              achieved_flops_per_s_per_gpu / 1e12)
            peak_tflops = omegaconf.OmegaConf.select(
              config, 'perf.theoretical_peak_tflops_per_gpu', default=None)
            if peak_tflops is not None and float(peak_tflops) > 0:
              metric_payload['perf/mfu'] = (
                achieved_flops_per_s_per_gpu / (float(peak_tflops) * 1e12))
          if gathered_diagnostics:
            allocated = []
            reserved = []
            max_allocated = []
            max_reserved = []
            for stage_idx, stage_payload in enumerate(gathered_diagnostics):
              if not stage_payload:
                continue
              stage_metrics = stage_payload.get('mem', {})
              metric_payload[f'perf/cuda_memory_allocated_gb/stage_{stage_idx}'] = float(
                stage_metrics.get('cuda_memory_allocated_gb', 0.0))
              metric_payload[f'perf/cuda_memory_reserved_gb/stage_{stage_idx}'] = float(
                stage_metrics.get('cuda_memory_reserved_gb', 0.0))
              metric_payload[f'perf/max_cuda_memory_allocated_gb/stage_{stage_idx}'] = float(
                stage_metrics.get('max_cuda_memory_allocated_gb', 0.0))
              metric_payload[f'perf/max_cuda_memory_reserved_gb/stage_{stage_idx}'] = float(
                stage_metrics.get('max_cuda_memory_reserved_gb', 0.0))
              allocated.append(float(stage_metrics.get('cuda_memory_allocated_gb', 0.0)))
              reserved.append(float(stage_metrics.get('cuda_memory_reserved_gb', 0.0)))
              max_allocated.append(float(stage_metrics.get('max_cuda_memory_allocated_gb', 0.0)))
              max_reserved.append(float(stage_metrics.get('max_cuda_memory_reserved_gb', 0.0)))
            if allocated:
              metric_payload['perf/cuda_memory_allocated_gb'] = max(allocated)
              metric_payload['perf/cuda_memory_reserved_gb'] = max(reserved)
              metric_payload['perf/max_cuda_memory_allocated_gb'] = max(max_allocated)
              metric_payload['perf/max_cuda_memory_reserved_gb'] = max(max_reserved)
        logger.info(
          "1F1B step=%d loss=%.6f step_time=%.3fs",
          global_step,
          float(loss_value.item()),
          elapsed,
        )
        if wandb_run is not None:
          wandb_run.log(metric_payload, step=global_step)

    if checkpoint_every > 0 and global_step % checkpoint_every == 0:
      _save_checkpoint(config, rank, stage_module, optimizer, scheduler, global_step)

  _save_checkpoint(config, rank, stage_module, optimizer, scheduler, global_step)
  dist.barrier()
  if rank == 0:
    logger.info("1F1B pipeline training completed at step %d.", global_step)
    if wandb_run is not None:
      wandb_run.finish()
  dist.destroy_process_group()
