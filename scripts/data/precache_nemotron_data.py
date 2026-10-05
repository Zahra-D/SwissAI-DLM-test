#!/usr/bin/env python3
"""Force the nemotron-cc-pretok train+valid caching pipeline to run and
save to disk, without building a model or Trainer.

Run this once ahead of the real sweep so the one-time cache-build cost
doesn't burn GPU-node-hours across multiple nodes -- this is meant to be
launched on a single node with num_proc bumped to use all available CPUs
(see scripts/data/precache_nemotron_data.run).

Usage (from repo root, inside the uni-d2 container):
  venv-docker/bin/python scripts/data/precache_nemotron_data.py \
    data=nemotron-cc-pretok model=gidd_hf model.length=2048 \
    loader.num_workers=224
"""
import functools
import math
import operator
import os
from pathlib import Path

import hydra
import omegaconf
import torch

from discrete_diffusion.data import get_dataset, get_tokenizer
from discrete_diffusion import utils

CONFIG_PATH = (Path(__file__).resolve().parents[1] / 'configs').as_posix()


def _register_resolver(name, resolver):
  if omegaconf.OmegaConf.has_resolver(name):
    return
  omegaconf.OmegaConf.register_new_resolver(name, resolver)


# Same resolvers registered by discrete_diffusion.__main__, needed since
# config composition (e.g. loader.batch_size) references them.
_register_resolver('cwd', os.getcwd)
_register_resolver('device_count', torch.cuda.device_count)
_register_resolver('div_up', lambda x, y: (x + y - 1) // y)
_register_resolver(
  'mul',
  lambda *args: functools.reduce(operator.mul, args) if args else 1)
_register_resolver('sub', lambda x, y: x - y)
_register_resolver('sqrt', lambda x: math.sqrt(float(x)))


def _cache_split(config, tokenizer, split_name, mode, insert_eos,
                  insert_special, min_length_key, chunking_key, num_proc,
                  pack_num_proc, pack_writer_batch_size):
  return get_dataset(
    split_name,
    tokenizer,
    mode=mode,
    wrap=config.data.wrap,
    insert_eos=insert_eos,
    insert_special_tokens=insert_special,
    cache_dir=config.data.cache_dir,
    block_size=config.model.length,
    streaming=config.data.streaming,
    num_proc=num_proc,
    revision=config.data.get(f"{mode}_revision", None),
    min_length=config.data.get(min_length_key, config.data.get("min_length", 0)),
    chunking=config.data.get(chunking_key, config.data.get("chunking", "none")),
    pretok_tokens_column=config.data.get("pretok_tokens_column", None),
    pretok_local_dir=config.data.get("pretok_local_dir", None),
    pretok_pack_num_proc=pack_num_proc,
    pretok_pack_writer_batch_size=pack_writer_batch_size)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name='config')
def main(config):
  logger = utils.get_logger(__name__)
  tokenizer = get_tokenizer(config)
  train_num_proc = int(config.loader.num_workers)
  valid_num_proc = int(config.loader.get("valid_num_workers", 1))
  train_pack_num_proc = int(
    config.loader.get("pack_num_workers", train_num_proc))
  valid_pack_num_proc = int(
    config.loader.get("valid_pack_num_workers", valid_num_proc))
  pack_writer_batch_size = config.loader.get("pack_writer_batch_size", None)
  if pack_writer_batch_size is not None:
    pack_writer_batch_size = int(pack_writer_batch_size)
  block_size = config.model.length
  logger.info(
    "Pre-caching with prepare_num_proc=(train=%d, valid=%d), "
    "pack_num_proc=(train=%d, valid=%d), pack_writer_batch_size=%s, "
    "block_size=%d",
    train_num_proc, valid_num_proc,
    train_pack_num_proc, valid_pack_num_proc,
    pack_writer_batch_size, block_size)

  logger.info("Caching train split: %s", config.data.train)
  train_set = _cache_split(
    config, tokenizer, config.data.train, "train",
    config.data.insert_train_eos,
    getattr(config.data, "insert_train_special", True),
    "train_min_length", "train_chunking", train_num_proc,
    train_pack_num_proc, pack_writer_batch_size)

  validation_split = (
    "test" if config.data.valid in ["text8", "lm1b", "ag_news"]
    else "validation")
  logger.info("Caching valid split: %s", config.data.valid)
  valid_set = _cache_split(
    config, tokenizer, config.data.valid, validation_split,
    config.data.insert_valid_eos,
    getattr(config.data, "insert_valid_special", True),
    "valid_min_length", "valid_chunking", valid_num_proc,
    valid_pack_num_proc, pack_writer_batch_size)

  train_rows = len(train_set)
  valid_rows = len(valid_set)
  logger.info(
    "Pre-caching complete. cache_dir=%s\n"
    "  train: %d packed sequences x %d tokens = %d tokens\n"
    "  valid: %d packed sequences x %d tokens = %d tokens\n"
    "  total (train+valid): %d tokens",
    config.data.cache_dir,
    train_rows, block_size, train_rows * block_size,
    valid_rows, block_size, valid_rows * block_size,
    (train_rows + valid_rows) * block_size)


if __name__ == '__main__':
  main()
