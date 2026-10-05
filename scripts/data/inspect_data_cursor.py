import sys

import torch


checkpoint = torch.load(
  sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
found = []
for key, value in checkpoint.items():
  if isinstance(value, dict) and "train_sampler" in value:
    found.append((key, value["train_sampler"]))
if not found:
  raise RuntimeError("No checkpointed train_sampler state found.")
for key, state in found:
  print(f"checkpoint_component={key}")
  print(f"train_sampler_state={state}")
