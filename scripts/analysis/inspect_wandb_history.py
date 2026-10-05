#!/usr/bin/env python3
"""Print selected scalar history rows from a local W&B .wandb record file."""

import json
import sys

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def _record_from_item(item):
  if isinstance(item, bytes):
    record = wandb_internal_pb2.Record()
    record.ParseFromString(item)
    return record
  return item


def _value(item):
  try:
    return json.loads(item.value_json)
  except Exception:
    return item.value_json


def main(path):
  if path.startswith("api:"):
    run = wandb.Api().run(path[len("api:"):])
    keys = ("trainer/global_step", "trainer/loss", "train/elbo", "val/nll",
            "val/bpd", "trainer/lr/matrix", "_runtime")
    print("run", run.path)
    print(",".join(keys))
    for row in run.scan_history(page_size=1000):
      if "val/nll" in row or int(row.get("trainer/global_step", -1)) % 250 == 0:
        print(",".join(str(row.get(key, "")) for key in keys))
    return

  store = DataStore()
  store.open_for_scan(path)
  rows = []
  while True:
    item = store.scan_data()
    if item is None:
      break
    record = _record_from_item(item)
    if not record.history:
      continue
    row = {entry.key: _value(entry) for entry in record.history.item}
    rows.append(row)

  keys = ("trainer/global_step", "trainer/loss", "train/elbo", "val/nll",
          "val/bpd", "trainer/lr/matrix", "_runtime")
  print("rows", len(rows))
  print("keys", sorted({key for row in rows for key in row})[:200])
  print(",".join(keys))
  for row in rows:
    if "val/nll" in row or int(row.get("trainer/global_step", -1)) % 250 == 0:
      print(",".join(str(row.get(key, "")) for key in keys))


if __name__ == "__main__":
  main(sys.argv[1])
