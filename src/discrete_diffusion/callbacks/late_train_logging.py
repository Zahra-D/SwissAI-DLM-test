"""Opt-in dense train-metric logging during a final optimizer-step window."""

from __future__ import annotations

import logging

import lightning as L


LOGGER = logging.getLogger(__name__)


class LateTrainLoggingCallback(L.Callback):
  """Write a dense final-window copy of a metric without changing its cadence.

  The ordinary ``train/elbo`` series remains at ``trainer.log_every_n_steps``.
  The final-window series is written directly to the experiment logger under a
  separate key, once per *optimizer* step.  This also works with gradient
  accumulation: repeated microbatches at a common global step are ignored.
  """

  def __init__(
      self,
      enabled: bool = False,
      last_n_steps: int = 100,
      source_metric_name: str = "train/elbo",
      output_metric_name: str = "analysis/train_elbo_final_window",
  ) -> None:
    super().__init__()
    if int(last_n_steps) <= 0:
      raise ValueError("last_n_steps must be positive")
    self.enabled = bool(enabled)
    self.last_n_steps = int(last_n_steps)
    self.source_metric_name = str(source_metric_name)
    self.output_metric_name = str(output_metric_name)
    self._last_logged_step = -1

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    del outputs, batch
    if not self.enabled:
      return
    # Lightning calls this hook for every microbatch.  Only the final
    # microbatch represents a completed optimizer update and should create a
    # final-window point.
    accumulation_steps = max(1, int(
        getattr(trainer, "accumulate_grad_batches", 1) or 1))
    if (int(batch_idx) + 1) % accumulation_steps != 0:
      return
    max_steps = int(getattr(trainer, "max_steps", 0) or 0)
    if max_steps <= 0:
      raise RuntimeError(
          "LateTrainLoggingCallback requires a finite trainer.max_steps")
    start_step = max(0, max_steps - self.last_n_steps)
    global_step = int(trainer.global_step)
    if global_step < start_step or global_step <= self._last_logged_step:
      return
    value = trainer.callback_metrics.get(self.source_metric_name)
    if value is None:
      return
    is_first_final_window_point = self._last_logged_step < 0
    self._last_logged_step = global_step
    if not getattr(trainer, "is_global_zero", False):
      return

    if hasattr(value, "detach"):
      value = value.detach().cpu().item()
    value = float(value)
    metrics = {self.output_metric_name: value}
    for logger in trainer.loggers:
      logger.log_metrics(metrics, step=global_step)
    if is_first_final_window_point:
      LOGGER.info(
          "[LATE_TRAIN_LOGGING] starting at global_step=%d: writing %s "
          "every optimizer step for the final %d updates; %s keeps its "
          "normal cadence",
          global_step, self.output_metric_name, self.last_n_steps,
          self.source_metric_name)
