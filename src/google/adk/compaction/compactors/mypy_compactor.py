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
import re

from ...events.event import Event
from ..models import CompactionStats
from ..models import FileLineRef
from ..models import Provenance
from ..models import ToolRunCompaction
from .base import BaseToolRunCompactor
from .generic import _cap_lines_by_token_budget
from .generic import _estimate_token_count
from .generic import _extract_command
from .generic import _extract_exit_code
from .generic import _extract_response_payload
from .generic import _safe_to_int
from .generic import _safe_to_text

_DEFAULT_TOKEN_BUDGET = 400
_MYPY_ERROR_PATTERN = re.compile(
    r'^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s+'
    r'error:\s+(?P<message>.+?)\s+\[(?P<code>[^\]]+)\]\s*$'
)


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


def _compact_text_for_stats(
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
) -> tuple[list[str], list[str], list[str], list[str]]:
  """Trims compacted sections until token estimate fits budget."""
  if token_budget <= 0:
    return [], [], [], []

  signatures = list(error_signatures)
  errors = list(key_errors)
  trace = list(trimmed_trace)
  snippets = list(salient_snippets)

  while (
      _estimate_token_count(
          _compact_text_for_stats(signatures, errors, trace, snippets)
      )
      > token_budget
  ):
    if snippets:
      snippets.pop(0)
      continue
    if trace:
      trace.pop(0)
      continue
    if errors:
      errors.pop(0)
      continue
    if signatures:
      signatures.pop(0)
      continue
    break

  return signatures, errors, trace, snippets


class MypyCompactor(BaseToolRunCompactor):
  """Compactor specialized for mypy diagnostics output."""

  def __init__(self, *, token_budget: int = _DEFAULT_TOKEN_BUDGET):
    self._token_budget = max(0, token_budget)

  def compact(self, event: Event) -> ToolRunCompaction | None:
    function_name, payload = _extract_response_payload(event)
    if not payload:
      return None

    command = _extract_command(function_name, payload)
    exit_code = _extract_exit_code(payload)
    stdout_text = _safe_to_text(payload.get('stdout'))
    stderr_text = _safe_to_text(payload.get('stderr'))
    output_lines = '\n'.join([stdout_text, stderr_text]).splitlines()

    error_signatures: list[str] = []
    key_errors: list[str] = []
    file_line_refs: list[FileLineRef] = []
    trimmed_source_lines: list[str] = []
    seen_refs: set[tuple[str, int, int]] = set()

    for line in output_lines:
      match = _MYPY_ERROR_PATTERN.match(line.strip())
      if match is None:
        continue
      path = match.group('path')
      line_num = int(match.group('line'))
      col_num = int(match.group('col'))
      message = match.group('message').strip()
      code = match.group('code').strip()

      error_signatures.append(f'mypy::{code}::{path}')
      key_errors.append(message)
      trimmed_source_lines.append(line.strip())

      ref_key = (path, line_num, col_num)
      if ref_key in seen_refs:
        continue
      seen_refs.add(ref_key)
      file_line_refs.append(FileLineRef(path=path, line=line_num, col=col_num))

    error_signatures = _dedupe_preserving_order(error_signatures)
    key_errors = _dedupe_preserving_order(key_errors)
    trimmed_trace = _cap_lines_by_token_budget(
        trimmed_source_lines,
        token_budget=self._token_budget,
    )
    salient_snippets = _cap_lines_by_token_budget(
        [*error_signatures, *key_errors],
        token_budget=max(1, self._token_budget // 4),
    )

    error_signatures, key_errors, trimmed_trace, salient_snippets = (
        _enforce_token_budget(
            error_signatures=error_signatures,
            key_errors=key_errors,
            trimmed_trace=trimmed_trace,
            salient_snippets=salient_snippets,
            token_budget=self._token_budget,
        )
    )

    compact_text = _compact_text_for_stats(
        error_signatures,
        key_errors,
        trimmed_trace,
        salient_snippets,
    )
    raw_text = '\n'.join([
        command,
        stdout_text,
        stderr_text,
        json.dumps(payload, sort_keys=True),
    ])
    raw_tokens_est = _estimate_token_count(raw_text)
    compact_tokens_est = _estimate_token_count(compact_text)
    if compact_tokens_est <= 0:
      compression_ratio = float(raw_tokens_est) if raw_tokens_est else 1.0
    else:
      compression_ratio = raw_tokens_est / compact_tokens_est

    return ToolRunCompaction(
        event_id=event.id,
        compaction_version=1,
        command=command,
        exit_code=exit_code,
        duration_ms=_safe_to_int(payload.get('duration_ms')),
        error_signatures=error_signatures,
        key_errors=key_errors,
        tests_failed=[],
        file_line_refs=file_line_refs,
        trimmed_trace=trimmed_trace,
        salient_snippets=salient_snippets,
        stats=CompactionStats(
            raw_tokens_est=raw_tokens_est,
            compact_tokens_est=compact_tokens_est,
            compression_ratio=compression_ratio,
        ),
        provenance=Provenance(event_id=event.id),
    )
