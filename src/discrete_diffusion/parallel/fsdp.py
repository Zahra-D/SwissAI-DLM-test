"""FSDP strategy definitions specific to the GIDD transformer."""

from __future__ import annotations

from typing import Any

from lightning.pytorch.strategies import FSDPStrategy

from ..models.modeling_gidd_hf import GiddLayer


class GiddFSDPStrategy(FSDPStrategy):
  """Full-shard GIDD at transformer-block granularity.

  A root-only FSDP wrapper must materialize all 8B parameters at once for the
  backward pass. Wrapping each :class:`GiddLayer` lets FSDP materialize one
  transformer block at a time instead.
  """

  def __init__(self, *args: Any, auto_wrap_policy=None, **kwargs: Any):
    if auto_wrap_policy is None:
      auto_wrap_policy = {GiddLayer}
    super().__init__(*args, auto_wrap_policy=auto_wrap_policy, **kwargs)

