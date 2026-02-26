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

import re

from ...events.event import Event
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
from .helpers import _build_compact_text_for_stats
from .helpers import _build_compaction_stats
from .helpers import _build_raw_text
from .helpers import _dedupe_preserving_order
from .helpers import _enforce_token_budget

_DEFAULT_TOKEN_BUDGET = 400
_MYPY_ERROR_PATTERN = re.compile(
    r'^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s+'
    r'error:\s+(?P<message>.+?)\s+\[(?P<code>[^\]]+)\]\s*$'
)


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
            estimate_token_count=_estimate_token_count,
        )
    )

    compact_text = _build_compact_text_for_stats(
        error_signatures=error_signatures,
        key_errors=key_errors,
        trimmed_trace=trimmed_trace,
        salient_snippets=salient_snippets,
    )
    raw_text = _build_raw_text(
        command=command,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        payload=payload,
    )
    stats = _build_compaction_stats(
        raw_text=raw_text,
        compact_text=compact_text,
        estimate_token_count=_estimate_token_count,
    )

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
        stats=stats,
        provenance=Provenance(event_id=event.id),
    )
