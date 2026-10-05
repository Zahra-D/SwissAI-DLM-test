"""Regression test for the path-only pickling contract used by spawn."""

from __future__ import annotations

import pickle
import sys

from discrete_diffusion.data.lazy_dataset import LazyDiskDataset


def main() -> int:
  wrapper = LazyDiskDataset('/a/path/that-is-not-opened-for-this-test', 17)
  wrapper._dataset = object()  # prove loaded state is stripped before pickle
  restored = pickle.loads(pickle.dumps(wrapper))
  assert len(restored) == 17
  assert restored.cache_path == '/a/path/that-is-not-opened-for-this-test'
  assert restored._dataset is None
  print('lazy spawn dataset pickling test passed')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
