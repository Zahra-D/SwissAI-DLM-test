"""FlexMDM-style recursive chunking utilities.

This module mirrors the delimiter-aware splitting logic that FlexMDM uses for
OpenWebText preprocessing so that the src/ loader can reproduce the same
document boundaries without re-implementing the entire dataset stack.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from transformers import PreTrainedTokenizerBase

__all__ = [
    "find_delimiter_positions",
    "recursive_split",
    "chunk_documents",
]


def find_delimiter_positions(tokens: Sequence[int], delimiter_tokens: Sequence[int]) -> List[int]:
  """Return all start indices where the delimiter occurs inside ``tokens``."""
  if not delimiter_tokens:
    return []
  matches: List[int] = []
  window = len(delimiter_tokens)
  needle = list(delimiter_tokens)
  for idx in range(len(tokens) - window + 1):
    if tokens[idx : idx + window] == needle:
      matches.append(idx)
  return matches


def recursive_split(tokens: Sequence[int], max_length: int, delimiter_tokens: Sequence[int]) -> List[List[int]]:
  """Recursively split ``tokens`` into ``max_length`` chunks along delimiters."""
  if len(tokens) <= max_length:
    return [list(tokens)]

  candidates = find_delimiter_positions(tokens, delimiter_tokens)
  if not candidates:
    return [list(tokens[i : i + max_length]) for i in range(0, len(tokens), max_length)]

  midpoint = len(tokens) // 2
  split_point = min(candidates, key=lambda pos: abs(pos - midpoint))
  delimiter_len = len(delimiter_tokens)

  left = recursive_split(tokens[:split_point], max_length, delimiter_tokens)
  right = recursive_split(tokens[split_point + delimiter_len :], max_length, delimiter_tokens)
  return left + right


def chunk_documents(tokenizer: PreTrainedTokenizerBase,
                    texts: Iterable[str],
                    max_length: int,
                    delimiter_tokens: Sequence[int],
                    *,
                    add_special_tokens: bool = False) -> dict:
  """Tokenize ``texts`` and split them into padded ``max_length`` chunks.

  Returns a HuggingFace-ready dict with ``input_ids``, ``attention_mask``,
  ``token_type_ids``, and the pre-padding ``length`` for each produced chunk.
  """

  pad_token = tokenizer.pad_token_id
  if pad_token is None:
    raise ValueError("Tokenizer must define a pad_token when using chunk_documents().")

  input_ids: List[List[int]] = []
  attn_masks: List[List[int]] = []
  token_type_ids: List[List[int]] = []
  lengths: List[int] = []

  for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    chunks = recursive_split(token_ids, max_length, delimiter_tokens)
    for chunk in chunks:
      if len(chunk) > max_length:
        raise ValueError("Chunk size exceeded max_length; recursive_split should prevent this.")
      pad_len = max_length - len(chunk)
      padded_chunk = list(chunk) + [pad_token] * pad_len
      mask = [1] * len(chunk) + [0] * pad_len
      input_ids.append(padded_chunk)
      attn_masks.append(mask)
      token_type_ids.append([0] * max_length)
      lengths.append(len(chunk))

  return {
      "input_ids": input_ids,
      "attention_mask": attn_masks,
      "token_type_ids": token_type_ids,
      "length": lengths,
  }
