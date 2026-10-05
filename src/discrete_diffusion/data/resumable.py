"""Exact-resume data utilities for map-style distributed training."""

from __future__ import annotations

import math
from typing import Any, Iterator, Optional, Sized

import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler


class StatefulDistributedSampler(Sampler[int]):
  """Distributed sampler whose per-rank cursor is checkpointable.

  The state is valid only when dataset length, world size, rank, batch size,
  seed, and drop-last policy are unchanged.  We fail loudly on a mismatch
  instead of silently changing the training data order after a resume.
  """

  def __init__(
      self,
      dataset: Sized,
      *,
      num_replicas: Optional[int] = None,
      rank: Optional[int] = None,
      shuffle: bool = True,
      seed: int = 0,
      drop_last: bool = False,
      batch_size: int,
  ) -> None:
    if num_replicas is None:
      num_replicas = dist.get_world_size() if dist.is_initialized() else 1
    if rank is None:
      rank = dist.get_rank() if dist.is_initialized() else 0
    if rank < 0 or rank >= num_replicas:
      raise ValueError(f"Invalid rank={rank} for num_replicas={num_replicas}.")
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")

    self.dataset = dataset
    self.num_replicas = int(num_replicas)
    self.rank = int(rank)
    self.shuffle = bool(shuffle)
    self.seed = int(seed)
    self.drop_last = bool(drop_last)
    self.batch_size = int(batch_size)
    self.epoch = 0
    self.position = 0

    dataset_size = len(self.dataset)
    if self.drop_last and dataset_size % self.num_replicas != 0:
      self.num_samples = math.ceil(
        (dataset_size - self.num_replicas) / self.num_replicas)
    else:
      self.num_samples = math.ceil(dataset_size / self.num_replicas)
    self.total_size = self.num_samples * self.num_replicas

  def _rank_indices(self) -> list[int]:
    dataset_size = len(self.dataset)
    if self.shuffle:
      generator = torch.Generator()
      generator.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(dataset_size, generator=generator).tolist()
    else:
      indices = list(range(dataset_size))

    if not self.drop_last:
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    else:
      indices = indices[:self.total_size]

    return indices[self.rank:self.total_size:self.num_replicas]

  def __iter__(self) -> Iterator[int]:
    indices = self._rank_indices()
    if self.position < 0 or self.position > len(indices):
      raise RuntimeError(
        f"Invalid sampler position={self.position}; rank has {len(indices)} samples.")
    for index in indices[self.position:]:
      self.position += 1
      yield index

  def __len__(self) -> int:
    return self.num_samples

  def set_epoch(self, epoch: int) -> None:
    epoch = int(epoch)
    if epoch != self.epoch:
      self.epoch = epoch
      self.position = 0

  def state_dict(self) -> dict[str, Any]:
    return {
      "version": 1,
      "epoch": self.epoch,
      "position": self.position,
      "dataset_size": len(self.dataset),
      "num_replicas": self.num_replicas,
      "shuffle": self.shuffle,
      "seed": self.seed,
      "drop_last": self.drop_last,
      "batch_size": self.batch_size,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    expected = {
      "version": 1,
      "dataset_size": len(self.dataset),
      "num_replicas": self.num_replicas,
      "shuffle": self.shuffle,
      "seed": self.seed,
      "drop_last": self.drop_last,
      "batch_size": self.batch_size,
    }
    mismatches = {
      key: (state.get(key), value)
      for key, value in expected.items()
      if state.get(key) != value
    }
    if mismatches:
      raise RuntimeError(
        "Cannot exactly resume the data order because sampler settings changed: "
        f"{mismatches}")
    self.epoch = int(state["epoch"])
    self.position = int(state["position"])
    if self.position < 0 or self.position > self.num_samples:
      raise RuntimeError(
        f"Checkpoint sampler position={self.position} is outside "
        f"[0, {self.num_samples}].")


class ResumableLoaderDataModule(L.LightningDataModule):
  """Wrap existing loaders so Lightning checkpoints their sampler cursor."""

  def __init__(
      self,
      train_loader: DataLoader,
      valid_loader: Optional[DataLoader],
  ) -> None:
    super().__init__()
    self._train_loader = train_loader
    self._valid_loader = valid_loader

  def train_dataloader(self) -> DataLoader:
    return self._train_loader

  def val_dataloader(self) -> Optional[DataLoader]:
    return self._valid_loader

  def state_dict(self) -> dict[str, Any]:
    sampler = self._train_loader.sampler
    if not isinstance(sampler, StatefulDistributedSampler):
      raise RuntimeError(
        "Exact resume requested but the train loader has no stateful sampler.")
    return {"train_sampler": sampler.state_dict()}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    sampler = self._train_loader.sampler
    if not isinstance(sampler, StatefulDistributedSampler):
      raise RuntimeError(
        "Exact resume requested but the train loader has no stateful sampler.")
    sampler.load_state_dict(state_dict["train_sampler"])
