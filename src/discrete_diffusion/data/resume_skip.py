"""Resume a run on exactly the batches it would have seen next.

Without ``loader.exact_resume`` the train loader is a plain ``DataLoader`` and
Lightning injects a ``DistributedSampler`` for it.  Lightning 2.5 restores its
batch counters from a checkpoint but does *not* fast-forward such a loader: it
warns that the loader "is not resumable" and starts the epoch's index order
from the beginning again, so a resumed run re-trains on data it has already
seen.

The injected sampler is ``DistributedSampler(dataset, num_replicas=world_size,
rank=global_rank, shuffle=True, seed=PL_GLOBAL_SEED, drop_last=False)`` and
its order depends only on those arguments and the epoch.  ``SkipDistributedSampler``
rebuilds that exact order and starts it at the position the checkpoint had
reached, so the first resumed micro-batch is the one the uninterrupted run
would have trained on next.
"""

from __future__ import annotations

import itertools
import logging
import os

import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler


LOGGER = logging.getLogger(__name__)

_LOADER_ATTRS_TO_KEEP = ("tokenizer", "_scion_trace_dataloader")


class SkipDistributedSampler(DistributedSampler):
  """``DistributedSampler`` that starts ``skip_samples`` into one epoch.

  For every epoch other than ``skip_epoch`` it is identical to
  ``DistributedSampler``.  ``__len__`` still reports the full per-rank epoch
  length: Lightning's restored batch counter resumes at the skipped position,
  so it reaches the end of the epoch exactly when this iterator runs out.
  """

  def __init__(self, dataset, *, num_replicas: int, rank: int, seed: int,
               skip_epoch: int, skip_samples: int, shuffle: bool = True,
               drop_last: bool = False) -> None:
    super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                     shuffle=shuffle, seed=seed, drop_last=drop_last)
    if not 0 <= skip_samples <= self.num_samples:
      raise ValueError(
          f"skip_samples={skip_samples} is outside this rank's epoch of "
          f"{self.num_samples} samples.")
    self.skip_epoch = int(skip_epoch)
    self.skip_samples = int(skip_samples)

  def __iter__(self):
    indices = super().__iter__()
    if self.epoch == self.skip_epoch and self.skip_samples:
      return itertools.islice(indices, self.skip_samples, None)
    return indices


def read_resume_position(ckpt_path: str, accumulate_grad_batches: int):
  """Return ``(epoch, microbatches, global_step)`` of a checkpoint.

  ``microbatches`` counts the micro-batches consumed in ``epoch``.

  The micro-batch count is taken from the optimizer step counter, because a
  checkpoint can be written part-way through an accumulation window: those
  partially accumulated gradients are not in the checkpoint, so the run must
  redo that optimizer step from its first micro-batch.
  """
  checkpoint = torch.load(
      ckpt_path, map_location="cpu", mmap=True, weights_only=False)
  fit_loop = checkpoint["loops"]["fit_loop"]
  epoch = int(fit_loop["epoch_progress"]["current"]["completed"])
  steps = int(fit_loop["epoch_loop.automatic_optimization.optim_progress"]
              ["optimizer"]["step"]["current"]["completed"])
  batches = int(fit_loop["epoch_loop.batch_progress"]["current"]["completed"])
  global_step = int(checkpoint["global_step"])
  del checkpoint
  microbatches = steps * int(accumulate_grad_batches)
  if batches != microbatches:
    # TrainerBase.on_save_checkpoint rewrites the batch counter to
    # optimizer_steps * accumulation; anything else means Lightning will
    # resume at a different position than the one we skip to.
    raise RuntimeError(
        f"Checkpoint {ckpt_path} records {batches} completed micro-batches "
        f"but {steps} optimizer steps x {accumulate_grad_batches} = "
        f"{microbatches}; cannot align the data position.")
  return epoch, microbatches, global_step


def with_resume_skip(loader: DataLoader, *, num_replicas: int, rank: int,
                     epoch: int, skip_batches: int) -> DataLoader:
  """Rebuild ``loader`` so it starts ``skip_batches`` into ``epoch``."""
  if isinstance(loader.sampler, DistributedSampler):
    raise ValueError(
        "The train loader already has a DistributedSampler; resume skipping "
        "only reproduces the sampler Lightning injects automatically.")
  if not isinstance(loader.sampler, RandomSampler):
    raise ValueError(
        "Resume skipping expects a shuffled train loader (RandomSampler), "
        f"got {type(loader.sampler).__name__}.")
  seed = os.environ.get("PL_GLOBAL_SEED")
  if seed is None:
    raise RuntimeError(
        "PL_GLOBAL_SEED is unset; Lightning seeds its injected sampler with "
        "it, so the original data order cannot be reproduced.")
  sampler = SkipDistributedSampler(
      loader.dataset, num_replicas=num_replicas, rank=rank, seed=int(seed),
      skip_epoch=epoch, skip_samples=skip_batches * loader.batch_size)
  kwargs = dict(
      batch_size=loader.batch_size, sampler=sampler,
      num_workers=loader.num_workers, collate_fn=loader.collate_fn,
      pin_memory=loader.pin_memory, drop_last=loader.drop_last,
      timeout=loader.timeout, worker_init_fn=loader.worker_init_fn,
      multiprocessing_context=loader.multiprocessing_context,
      generator=loader.generator, prefetch_factor=loader.prefetch_factor,
      persistent_workers=loader.persistent_workers,
      pin_memory_device=loader.pin_memory_device)
  if hasattr(loader, "in_order"):
    kwargs["in_order"] = loader.in_order
  resumed = DataLoader(loader.dataset, **kwargs)
  for attr in _LOADER_ATTRS_TO_KEEP:
    if hasattr(loader, attr):
      setattr(resumed, attr, getattr(loader, attr))
  LOGGER.info(
      "[RESUME_SKIP] rank %d/%d: epoch %d starts %d micro-batches (%d "
      "samples) into the seed-%s DistributedSampler order.",
      rank, num_replicas, epoch, skip_batches,
      skip_batches * loader.batch_size, seed)
  return resumed


class ResumeAlignmentCheck(L.Callback):
  """Fail before the first resumed step if Lightning resumes elsewhere.

  The skip above assumes Lightning's first resumed ``batch_idx`` equals the
  number of skipped micro-batches and begins an accumulation window.  If
  either assumption breaks, training would silently pair the wrong data with
  the wrong optimizer step, so stop instead.
  """

  def __init__(self, expected_batch_idx: int, expected_global_step: int):
    super().__init__()
    self.expected_batch_idx = int(expected_batch_idx)
    self.expected_global_step = int(expected_global_step)
    self._checked = False

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if self._checked:
      return
    self._checked = True
    accumulation = max(1, int(trainer.accumulate_grad_batches or 1))
    if (int(batch_idx) != self.expected_batch_idx
        or int(trainer.global_step) != self.expected_global_step
        or int(batch_idx) % accumulation != 0):
      raise RuntimeError(
          f"[RESUME_SKIP] Lightning resumed at batch_idx={batch_idx}, "
          f"global_step={trainer.global_step}; expected batch_idx="
          f"{self.expected_batch_idx}, global_step="
          f"{self.expected_global_step} at an accumulation boundary.")
    LOGGER.info(
        "[RESUME_SKIP] first resumed micro-batch: batch_idx=%d, "
        "global_step=%d (aligned).", batch_idx, trainer.global_step)
