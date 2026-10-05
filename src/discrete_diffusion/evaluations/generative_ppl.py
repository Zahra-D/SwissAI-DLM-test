"""Standalone Generative Perplexity evaluation.

Loads generated samples and evaluates NLL/PPL with a chosen eval LM.

Supported sample formats:
- .pt: torch.Tensor of token ids (shape [N, T] or [N, 1, T])
- .npz: numpy array under key 'samples' (shape [N, T])
- .json: contains base64-encoded numpy array under key 'np_tokens_b64'

This script decodes using `model_tokenizer` and (optionally) retokenizes with
the eval model's tokenizer before computing loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_samples(samples_path: str) -> np.ndarray:
  path = Path(hydra.utils.to_absolute_path(samples_path))
  if not path.exists():
    raise FileNotFoundError(f"Samples not found at {path}")

  if path.suffix == ".pt":
    z_ts = torch.load(path, weights_only=True)
    if isinstance(z_ts, torch.Tensor):
      arr = z_ts.detach().cpu()
    else:
      # if saved as dict or list, try common keys/shapes
      raise ValueError("Unsupported .pt structure; expected a Tensor.")
    if arr.ndim == 3 and arr.shape[1] == 1:
      arr = arr.squeeze(1)
    return arr.numpy()

  if path.suffix == ".npz":
    content = np.load(path)
    if 'samples' not in content:
      raise KeyError(".npz must contain 'samples' key")
    return content['samples']

  if path.suffix == ".json":
    from ..utils import utils as _utils
    with open(path, 'r') as f:
      payload = json.load(f)
    if 'np_tokens_b64' not in payload:
      raise KeyError(".json must contain 'np_tokens_b64' key")
    arr = _utils.base64_to_np(payload['np_tokens_b64'])
    return arr

  raise ValueError(f"Unsupported samples format: {path.suffix}")


def _retokenize(
  texts: List[str],
  tokenizer: AutoTokenizer,
  max_length: int,
  device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  # Default context windows for common models; conservative fallback
  eval_context_size = 4096 if 'llama' in tokenizer.name_or_path.lower() else 1024
  batch = tokenizer(
    texts,
    return_tensors="pt",
    return_token_type_ids=False,
    return_attention_mask=True,
    truncation=True,
    padding=True,
    max_length=max_length,
  )
  attn_mask = batch['attention_mask'].to(device)
  input_ids = batch['input_ids'].to(device)
  return input_ids, attn_mask, eval_context_size


@hydra.main(config_path='../../../configs/eval', config_name='gen_ppl', version_base='1.3')
def main(cfg):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_float32_matmul_precision('high')
  torch.set_grad_enabled(False)

  # Decode tokens (from diffusion model) to text using its tokenizer
  model_tokenizer = AutoTokenizer.from_pretrained(cfg.model_tokenizer)

  eval_model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_model, device_map="auto")
  eval_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
  if eval_tokenizer.pad_token_id is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

  if cfg.torch_compile:
    eval_model = torch.compile(eval_model)

  # Load samples and make text
  z_ts = _load_samples(cfg.samples_path)
  if z_ts.ndim != 2:
    raise ValueError(f"Expected 2D [N, T] tokens array, got {z_ts.shape}")
  texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)

  total_acc = 0.0
  total_nll = 0.0
  total_tokens = 0.0
  all_nlls: List[float] = []

  with torch.no_grad():
    for i in range(0, len(texts), cfg.batch_size):
      xs = texts[i:i + cfg.batch_size]

      if cfg.retokenize:
        input_ids, attn_mask, context_size = _retokenize(
          xs, eval_tokenizer, cfg.max_length, device)
      else:
        # Use model tokens directly (not recommended across tokenizers)
        # Here we re-tokenize anyway but with the same tokenizer to ensure tensors
        input_ids, attn_mask, context_size = _retokenize(
          xs, eval_tokenizer, cfg.max_length, device)

      # Evaluate possibly only the first chunk up to EOS
      logits = eval_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, :-1]
      labels = input_ids[:, 1:]
      loss_mask = attn_mask[:, :-1]

      nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)

      if cfg.first_chunk_only:
        eos_id = eval_tokenizer.eos_token_id
        eos_mask = (labels == eos_id).cumsum(-1) == 0  # valid until first EOS (exclusive)
        # Ensure we still respect attention mask
        valid = loss_mask.bool() & eos_mask
      else:
        valid = loss_mask.bool()

      valid = valid.to(nll.dtype)
      all_nlls.extend(nll[valid == 1].detach().cpu().numpy().tolist())
      total_nll += float((nll * valid).sum().item())

      acc = (logits.argmax(-1) == labels).to(nll.dtype)
      total_acc += float((acc * valid).sum().item())
      total_tokens += float(valid.sum().item())

  if total_tokens == 0:
    raise RuntimeError("No valid tokens for evaluation (check inputs/EOS handling)")

  avg_nll = total_nll / total_tokens
  ppl = float(np.exp(avg_nll))
  acc = total_acc / total_tokens

  metrics = {
    "file": Path(cfg.samples_path).stem,
    "pretrained_model": cfg.pretrained_model,
    "median_nll": float(np.median(all_nlls)) if all_nlls else float('nan'),
    "avg_nll": float(avg_nll),
    "ppl": float(ppl),
    "acc": float(acc),
    "tokens": int(total_tokens),
    "retokenize": bool(cfg.retokenize),
    "first_chunk_only": bool(cfg.first_chunk_only),
  }

  print(json.dumps(metrics, indent=2))
  out_path = Path(hydra.utils.to_absolute_path(cfg.metrics_path))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, 'w') as f:
    json.dump(metrics, f)
  print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
  main()

