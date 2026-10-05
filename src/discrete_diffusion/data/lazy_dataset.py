"""Small, spawn-friendly wrapper for a HuggingFace dataset saved on disk.

``datasets.Dataset`` contains a large Arrow/shard metadata graph.  Passing a
loaded instance to a ``spawn`` DataLoader serializes that graph once per
worker.  This wrapper deliberately serializes only the on-disk path and the
dataset length.  Each worker opens its own memory-mapped Arrow handles lazily
when it receives its first index.

This improves worker startup but does *not* make a multi-worker loader exactly
resumable: PyTorch prefetches samples in worker queues.
"""

from __future__ import annotations

from typing import Any

import datasets
import torch


class LazyDiskDataset(torch.utils.data.Dataset):
  """Map-style dataset that reopens a HF Arrow cache separately per process."""

  def __init__(self, cache_path: str, length: int):
    self.cache_path = str(cache_path)
    self._length = int(length)
    # This is intentionally absent from the pickled state sent to workers.
    self._dataset: datasets.Dataset | None = None

  def __len__(self) -> int:
    return self._length

  def _get_dataset(self) -> datasets.Dataset:
    if self._dataset is None:
      self._dataset = datasets.load_from_disk(self.cache_path).with_format("torch")
    return self._dataset

  def __getitem__(self, index: int) -> Any:
    return self._get_dataset()[index]

  def __getstate__(self):
    # Be defensive if the parent happened to access an item before spawning.
    # Arrow tables, indices, and file mappings must never travel through the
    # multiprocessing pipe.
    return {"cache_path": self.cache_path, "_length": self._length,
            "_dataset": None}
