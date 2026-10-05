from discrete_diffusion.data.resumable import StatefulDistributedSampler


class DummyDataset:
  def __len__(self):
    return 100


dataset = DummyDataset()
world_size = 4
batch_size = 2

for rank in range(world_size):
  baseline_sampler = StatefulDistributedSampler(
    dataset, num_replicas=world_size, rank=rank, seed=17,
    batch_size=batch_size)
  baseline = list(baseline_sampler)
  interrupted_sampler = StatefulDistributedSampler(
    dataset, num_replicas=world_size, rank=rank, seed=17,
    batch_size=batch_size)
  interrupted_iter = iter(interrupted_sampler)
  prefix = [next(interrupted_iter) for _ in range(8)]
  state = interrupted_sampler.state_dict()
  resumed_sampler = StatefulDistributedSampler(
    dataset, num_replicas=world_size, rank=rank, seed=17,
    batch_size=batch_size)
  resumed_sampler.load_state_dict(state)
  assert prefix + list(resumed_sampler) == baseline

rank0 = StatefulDistributedSampler(
  dataset, num_replicas=world_size, rank=0, seed=17,
  batch_size=batch_size)
rank0_iter = iter(rank0)
for _ in range(8):
  next(rank0_iter)
rank0_state = rank0.state_dict()
rank3 = StatefulDistributedSampler(
  dataset, num_replicas=world_size, rank=3, seed=17,
  batch_size=batch_size)
rank3.load_state_dict(rank0_state)
rank3_baseline = list(StatefulDistributedSampler(
  dataset, num_replicas=world_size, rank=3, seed=17,
  batch_size=batch_size))
assert list(rank3) == rank3_baseline[8:]

shards = [set(StatefulDistributedSampler(
  dataset, num_replicas=world_size, rank=rank, seed=17,
  batch_size=batch_size)) for rank in range(world_size)]
assert set.union(*shards) == set(range(len(dataset)))
for i in range(world_size):
  for j in range(i + 1, world_size):
    assert shards[i].isdisjoint(shards[j])

# An auxiliary trace iterator may read batches, but restoring the sampler
# snapshot must leave the optimizer-training iterator and cursor unchanged.
trace_sampler = StatefulDistributedSampler(
  dataset, num_replicas=world_size, rank=0, seed=17,
  batch_size=batch_size)
main_iter = iter(trace_sampler)
main_prefix = [next(main_iter) for _ in range(batch_size)]
cursor_before_trace = trace_sampler.state_dict()
trace_iter = iter(trace_sampler)
_ = [next(trace_iter) for _ in range(3 * batch_size)]
trace_sampler.load_state_dict(cursor_before_trace)
main_next = [next(main_iter) for _ in range(batch_size)]
rank0_baseline = list(StatefulDistributedSampler(
  dataset, num_replicas=world_size, rank=0, seed=17,
  batch_size=batch_size))
assert main_prefix + main_next == rank0_baseline[:2 * batch_size]
assert trace_sampler.position == 2 * batch_size

print("resumable sampler sequence tests passed")
