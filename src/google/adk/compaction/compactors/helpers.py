# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from collections.abc import Callable

from ..models import CompactionStats


def _dedupe_preserving_order(items: list[str]) -> list[str]:
  """Deduplicates string items while preserving first-seen order."""
  deduped: list[str] = []
  seen: set[str] = set()
  for item in items:
    if item in seen:
      continue
    seen.add(item)
    deduped.append(item)
  return deduped


def _build_compact_text_for_stats(
    *,
    error_signatures: list[str],
    key_errors: list[str],
    trimmed_trace: list[str],
    salient_snippets: list[str],
) -> str:
  """Builds compact text payload used for token estimates."""
  return '\n'.join([
      '\n'.join(error_signatures),
      '\n'.join(key_errors),
      '\n'.join(trimmed_trace),
      '\n'.join(salient_snippets),
  ])


def _enforce_token_budget(
    *,
    error_signatures: list[str],
    key_errors: list[str],
    trimmed_trace: list[str],
    salient_snippets: list[str],
    token_budget: int,
    estimate_token_count: Callable[[str], int],
) -> tuple[list[str], list[str], list[str], list[str]]:
  """Trims compacted sections until token estimate fits budget."""
  if token_budget <= 0:
    return [], [], [], []

  signatures = list(error_signatures)
  errors = list(key_errors)
  trace = list(trimmed_trace)
  snippets = list(salient_snippets)

  while (
      estimate_token_count(
          _build_compact_text_for_stats(
              error_signatures=signatures,
              key_errors=errors,
              trimmed_trace=trace,
              salient_snippets=snippets,
          )
      )
      > token_budget
  ):
    if snippets:
      snippets.pop()
      continue
    if trace:
      trace.pop()
      continue
    if errors:
      errors.pop()
      continue
    if signatures:
      signatures.pop()
      continue
    break

  return signatures, errors, trace, snippets


def _build_raw_text(
    *,
    command: str,
    stdout_text: str,
    stderr_text: str,
    payload: object,
) -> str:
  """Builds raw text payload used for token estimates."""
  return '\n'.join([
      command,
      stdout_text,
      stderr_text,
      json.dumps(payload, sort_keys=True),
  ])


def _build_compaction_stats(
    *,
    raw_text: str,
    compact_text: str,
    estimate_token_count: Callable[[str], int],
) -> CompactionStats:
  """Builds token estimate statistics for compaction output."""
  raw_tokens_est = estimate_token_count(raw_text)
  compact_tokens_est = estimate_token_count(compact_text)

  if compact_tokens_est <= 0:
    compression_ratio = float(raw_tokens_est) if raw_tokens_est else 1.0
  else:
    compression_ratio = raw_tokens_est / compact_tokens_est

  return CompactionStats(
      raw_tokens_est=raw_tokens_est,
      compact_tokens_est=compact_tokens_est,
      compression_ratio=compression_ratio,
  )
