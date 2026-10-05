"""Opt-in PyTorch profiler callback for short training traces."""

from __future__ import annotations

import os
from typing import Optional

import lightning as L
import torch


class PyTorchProfilerCallback(L.Callback):
  """Capture a short PyTorch profiler trace after warmup.

  ``start_step`` is measured in optimizer steps. ``warmup_batches`` and
  ``active_batches`` are measured in Lightning training batches, which means
  microbatches when gradient accumulation is enabled.
  """

  def __init__(
      self,
      enabled: bool = False,
      start_step: int = 2,
      warmup_batches: int = 2,
      active_batches: int = 4,
      dirpath: str = "pytorch_profiler",
      filename: str = "profile",
      row_limit: int = 50,
      record_shapes: bool = True,
      profile_memory: bool = True,
      with_stack: bool = False,
      with_flops: bool = True,
      export_chrome_trace: bool = True,
      export_memory_timeline: bool = False,
      profile_all_ranks: bool = False,
      sort_by: str = "cuda_time_total") -> None:
    super().__init__()
    self.enabled = enabled
    self.start_step = start_step
    self.warmup_batches = warmup_batches
    self.active_batches = active_batches
    self.dirpath = dirpath
    self.filename = filename
    self.row_limit = row_limit
    self.record_shapes = record_shapes
    self.profile_memory = profile_memory
    self.with_stack = with_stack
    self.with_flops = with_flops
    self.export_chrome_trace = export_chrome_trace
    self.export_memory_timeline = export_memory_timeline
    self.profile_all_ranks = profile_all_ranks
    self.sort_by = sort_by
    self._profiler: Optional[torch.profiler.profile] = None
    self._started = False
    self._finished = False
    self._num_steps = 0

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if not self._should_profile(trainer):
      return
    if self._started or trainer.global_step < self.start_step:
      return

    os.makedirs(self.dirpath, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
      activities.append(torch.profiler.ProfilerActivity.CUDA)

    self._profiler = torch.profiler.profile(
      activities=activities,
      schedule=torch.profiler.schedule(
        wait=0,
        warmup=max(0, self.warmup_batches),
        active=max(1, self.active_batches),
        repeat=1),
      on_trace_ready=lambda prof: self._write_profile(trainer, prof),
      record_shapes=self.record_shapes,
      profile_memory=self.profile_memory,
      with_stack=self.with_stack,
      with_flops=self.with_flops)
    self._profiler.start()
    self._started = True
    self._num_steps = 0
    print(
      "Started PyTorch profiler: "
      f"rank={getattr(trainer, 'global_rank', 0)}, "
      f"global_step={trainer.global_step}, "
      f"warmup_batches={self.warmup_batches}, "
      f"active_batches={self.active_batches}",
      flush=True)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self._profiler is None or self._finished:
      return

    self._profiler.step()
    self._num_steps += 1
    if self._num_steps >= max(0, self.warmup_batches) + max(
        1, self.active_batches):
      self._stop()

  def on_fit_end(self, trainer, pl_module):
    self._stop()

  def on_exception(self, trainer, pl_module, exception):
    self._stop()

  def _should_profile(self, trainer) -> bool:
    if not self.enabled or self._finished:
      return False
    if self.profile_all_ranks:
      return True
    return int(getattr(trainer, "global_rank", 0) or 0) == 0

  def _write_profile(self, trainer, profiler):
    rank = int(getattr(trainer, "global_rank", 0) or 0)
    step = int(getattr(trainer, "global_step", 0) or 0)
    base = os.path.join(self.dirpath, f"{self.filename}_rank{rank}_step{step}")

    if self.export_chrome_trace:
      profiler.export_chrome_trace(f"{base}.json")

    table = self._profile_table(profiler)
    with open(f"{base}.txt", "w", encoding="utf-8") as handle:
      handle.write(table)
      handle.write("\n")

    if self.export_memory_timeline and self.profile_memory:
      try:
        profiler.export_memory_timeline(f"{base}_memory.html")
      except Exception as exc:  # pragma: no cover - profiler-version dependent.
        with open(f"{base}_memory_error.txt", "w", encoding="utf-8") as handle:
          handle.write(f"{type(exc).__name__}: {exc}\n")

  def _profile_table(self, profiler) -> str:
    key_averages = profiler.key_averages(
      group_by_input_shape=self.record_shapes)
    sort_candidates = (
      self.sort_by,
      "cuda_time_total",
      "self_cuda_time_total",
      "cpu_time_total",
      "self_cpu_time_total")
    last_error = None
    for sort_by in sort_candidates:
      try:
        return key_averages.table(sort_by=sort_by, row_limit=self.row_limit)
      except Exception as exc:  # pragma: no cover - profiler-version dependent.
        last_error = exc
    return f"Could not render profiler table: {last_error}"

  def _stop(self):
    if self._profiler is None:
      return
    try:
      self._profiler.stop()
    finally:
      self._profiler = None
      self._finished = True
