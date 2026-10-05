"""Save and restore every rank's random state with each checkpoint.

Lightning 2.5 checkpoints do not contain RNG state, so a resumed run draws
different diffusion noise (time lattice shift, token corruption) than the
uninterrupted run would have.  This callback makes the continuation draw the
same noise:

* Each rank writes its own Python / NumPy / torch CPU / torch CUDA state to
  ``<dirpath>/step<N>/rank<R>.pt`` whenever a checkpoint is written; the
  checkpoint itself records only which ``step<N>`` directory belongs to it.
  Writing per-rank files needs no collective, so it is safe even when a
  checkpoint is triggered from a signal handler.
* The saved state is the one at the optimizer-step boundary the run resumes
  from.  A checkpoint written part-way through an accumulation window gets
  the state captured at the start of that window, because resuming redoes
  the whole window.
* Validation runs on a copy of the random state and the training stream is
  put back afterwards.  Lightning skips the validation that falls on the
  checkpoint step when it resumes, so without this the resumed training
  stream would be offset by that validation's draws.

If a checkpoint has no saved state (older checkpoints), every generator is
reseeded from ``(PL_GLOBAL_SEED, global_step)`` so the continuation is at least
reproducible, and a warning says the noise differs from the original run.

``reseed_every_step`` instead reseeds every generator at the start of each
optimizer step from ``(PL_GLOBAL_SEED, global_step, global_rank)``.  The noise
of a step then depends only on the step and the rank, never on the history,
so runs that share seed and world size draw identical noise at every step
even when they were resumed at different steps.  This keeps the
common-random-numbers pairing of an lr sweep after resuming old checkpoints
that carry no saved state.  ``reseed_from_step`` delays it to later steps.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn


LOGGER = logging.getLogger(__name__)

_FORMAT = 1


def _capture():
  return {
      "python": random.getstate(),
      "numpy": np.random.get_state(),
      "torch": torch.get_rng_state(),
      "torch.cuda": (torch.cuda.get_rng_state()
                     if torch.cuda.is_available() else None),
  }


def _step_seed(base_seed, global_step, global_rank):
  key = f"{int(base_seed)}:{int(global_step)}:{int(global_rank)}".encode()
  return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def _restore(states):
  random.setstate(states["python"])
  np.random.set_state(states["numpy"])
  torch.set_rng_state(states["torch"])
  if states.get("torch.cuda") is not None and torch.cuda.is_available():
    torch.cuda.set_rng_state(states["torch.cuda"])


class RngStateCallback(L.Callback):

  def __init__(self, dirpath: str, enabled: bool = True,
               isolate_validation: bool = True,
               reseed_every_step: bool = False,
               reseed_from_step: int = 0) -> None:
    super().__init__()
    self.dirpath = str(dirpath)
    self.enabled = bool(enabled)
    self.isolate_validation = bool(isolate_validation)
    self.reseed_every_step = bool(reseed_every_step)
    self.reseed_from_step = int(reseed_from_step)
    self._base_seed = None
    self._reseed_logged = False
    self._trainer = None
    self._window_start = None  # (batch_idx, states) at an accumulation start
    self._validation_stash = None
    self._to_restore = None

  def setup(self, trainer, pl_module, stage):
    self._trainer = trainer

  # ------------------------------------------------------------------ capture
  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if not self.enabled:
      return
    accumulation = max(1, int(trainer.accumulate_grad_batches or 1))
    if int(batch_idx) % accumulation != 0:
      return
    if (self.reseed_every_step
        and int(trainer.global_step) >= self.reseed_from_step):
      if self._base_seed is None:
        self._base_seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
      seed = _step_seed(self._base_seed, trainer.global_step,
                        trainer.global_rank)
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if not self._reseed_logged:
        self._reseed_logged = True
        LOGGER.info(
            "[RNG_STATE] rank %d: reseeding every optimizer step from "
            "(PL_GLOBAL_SEED, global_step, rank), starting at step %d.",
            trainer.global_rank, trainer.global_step)
    self._window_start = (int(batch_idx), _capture())

  def on_validation_start(self, trainer, pl_module):
    if self.enabled and self.isolate_validation:
      self._validation_stash = _capture()

  def on_validation_end(self, trainer, pl_module):
    if self._validation_stash is not None:
      _restore(self._validation_stash)
      self._validation_stash = None

  def _boundary_states(self, trainer):
    """Training-stream state after the completed optimizer steps."""
    accumulation = max(1, int(trainer.accumulate_grad_batches or 1))
    optim_progress = (trainer.fit_loop.epoch_loop.automatic_optimization
                      .optim_progress)
    boundary = int(optim_progress.optimizer.step.current.completed) * accumulation
    if self._window_start is not None and self._window_start[0] == boundary:
      # Saved inside the window that starts at the boundary.
      return self._window_start[1], "accumulation-window start"
    if self._validation_stash is not None:
      # Saved during validation; the stash is the training stream.
      return self._validation_stash, "pre-validation"
    return _capture(), "current"

  def state_dict(self):
    trainer = self._trainer
    if (not self.enabled or trainer is None
        or trainer.state.fn != TrainerFn.FITTING):
      return {}
    states, source = self._boundary_states(trainer)
    tag = f"step{int(trainer.global_step):09d}"
    directory = Path(self.dirpath) / tag
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{int(trainer.global_rank):05d}.pt"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(states, tmp)
    os.replace(tmp, path)
    LOGGER.info("[RNG_STATE] rank %d saved %s state to %s",
                trainer.global_rank, source, path)
    return {"format": _FORMAT, "dirpath": self.dirpath, "tag": tag,
            "world_size": int(trainer.world_size),
            "global_step": int(trainer.global_step)}

  # ------------------------------------------------------------------ restore
  def load_state_dict(self, state_dict):
    self._to_restore = dict(state_dict) if state_dict else None

  def on_train_start(self, trainer, pl_module):
    if not self.enabled or int(trainer.global_step) == 0:
      return
    if (self.reseed_every_step
        and int(trainer.global_step) >= self.reseed_from_step):
      # The first resumed step reseeds from the step itself.
      self._to_restore = None
      return
    state = self._to_restore
    self._to_restore = None
    reason = None
    if not state:
      reason = "the checkpoint has no saved random state"
    elif int(state.get("world_size", -1)) != int(trainer.world_size):
      reason = (f"it was saved with world_size={state.get('world_size')}, "
                f"now {trainer.world_size}")
    elif int(state.get("global_step", -1)) != int(trainer.global_step):
      reason = (f"it belongs to step {state.get('global_step')}, resuming "
                f"at {trainer.global_step}")
    else:
      path = (Path(state["dirpath"]) / state["tag"]
              / f"rank{int(trainer.global_rank):05d}.pt")
      if not path.is_file():
        reason = f"{path} is missing"
      else:
        _restore(torch.load(path, map_location="cpu", weights_only=False))
        LOGGER.info("[RNG_STATE] rank %d restored random state from %s",
                    trainer.global_rank, path)
        return
    seed = (int(os.environ.get("PL_GLOBAL_SEED", 0)) * 1_000_003
            + int(trainer.global_step)) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    LOGGER.warning(
        "[RNG_STATE] rank %d cannot restore the saved random state because "
        "%s; reseeded every generator with %d (from PL_GLOBAL_SEED and "
        "global_step). Data order is unaffected, but diffusion noise differs "
        "from the original run from here on.",
        trainer.global_rank, reason, seed)
