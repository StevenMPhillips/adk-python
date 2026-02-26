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
from .generic import _extract_file_line_refs
from .generic import _extract_response_payload
from .generic import _safe_to_int
from .generic import _safe_to_text
from .helpers import _build_compact_text_for_stats
from .helpers import _build_compaction_stats
from .helpers import _build_raw_text
from .helpers import _dedupe_preserving_order

_DEFAULT_TOKEN_BUDGET = 600
_DEFAULT_ERROR_TYPE = 'UnknownError'
_FAILED_ERROR_LINE_PATTERN = re.compile(
    r'^(?:FAILED|ERROR)\s+(?P<test_id>\S+)' r'(?:\s*-\s*(?P<reason>.+))?$'
)
_TRACEBACK_FILE_LINE_PATTERN = re.compile(
    r'^\s*File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+)'
)
_ERROR_TYPE_PATTERN = re.compile(
    r'(?P<error_type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failed|Exit))'
)


def _extract_error_type(text: str) -> str | None:
  """Returns an error type name when one is present."""
  if not text:
    return None
  match = _ERROR_TYPE_PATTERN.search(text)
  if match is None:
    return None
  return match.group('error_type')


def _split_test_id(test_id: str) -> tuple[str, str]:
  """Splits a pytest node id into test path and test name."""
  if '::' not in test_id:
    return test_id, test_id
  test_path, test_name = test_id.split('::', 1)
  return test_path, test_name


def _extract_traceback_file_line_refs(lines: list[str]) -> list[FileLineRef]:
  """Extracts references from traceback-style `File` lines."""
  refs: list[FileLineRef] = []
  seen: set[tuple[str, int]] = set()
  for line in lines:
    match = _TRACEBACK_FILE_LINE_PATTERN.match(line)
    if match is None:
      continue
    path = match.group('path')
    line_num = int(match.group('line'))
    key = (path, line_num)
    if key in seen:
      continue
    seen.add(key)
    refs.append(FileLineRef(path=path, line=line_num, col=None))
  return refs


class PytestCompactor(BaseToolRunCompactor):
  """Compactor specialized for pytest output streams."""

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

    tests_failed: list[str] = []
    summary_error_type_by_test: dict[str, str] = {}
    assertion_messages: list[str] = []
    trace_lines: list[str] = []

    for line in output_lines:
      stripped_line = line.strip()

      summary_match = _FAILED_ERROR_LINE_PATTERN.match(stripped_line)
      if summary_match is not None:
        test_id = summary_match.group('test_id')
        tests_failed.append(test_id)
        reason = summary_match.group('reason') or ''
        reason_error_type = _extract_error_type(reason)
        if reason_error_type:
          summary_error_type_by_test[test_id] = reason_error_type
        trace_lines.append(stripped_line)
        continue

      if line.startswith('E   '):
        message = line[4:].strip()
        if message:
          assertion_messages.append(message)
        trace_lines.append(stripped_line)
        continue

      if stripped_line.startswith(
          'Traceback (most recent call last):'
      ) or _TRACEBACK_FILE_LINE_PATTERN.match(line):
        trace_lines.append(stripped_line)

    tests_failed = _dedupe_preserving_order(tests_failed)
    assertion_messages = _dedupe_preserving_order(assertion_messages)

    fallback_error_type = _DEFAULT_ERROR_TYPE
    for message in assertion_messages:
      message_error_type = _extract_error_type(message)
      if message_error_type:
        fallback_error_type = message_error_type
        break

    error_signatures: list[str] = []
    for test_id in tests_failed:
      test_path, test_name = _split_test_id(test_id)
      error_type = summary_error_type_by_test.get(test_id, fallback_error_type)
      signature = f'pytest::{error_type}::{test_path}::{test_name}'
      error_signatures.append(signature)

    error_signatures = _dedupe_preserving_order(error_signatures)

    key_errors = assertion_messages[:5]
    if not key_errors:
      key_errors = [line for line in trace_lines if line][:5]

    file_line_refs = _extract_file_line_refs(
        '\n'.join([stdout_text, stderr_text])
    )
    traceback_refs = _extract_traceback_file_line_refs(output_lines)
    seen_refs: set[tuple[str, int, int | None]] = {
        (ref.path, ref.line, ref.col) for ref in file_line_refs
    }
    for ref in traceback_refs:
      key = (ref.path, ref.line, ref.col)
      if key in seen_refs:
        continue
      seen_refs.add(key)
      file_line_refs.append(ref)

    trimmed_trace = _cap_lines_by_token_budget(
        trace_lines,
        token_budget=self._token_budget,
    )
    salient_snippets = _cap_lines_by_token_budget(
        [*error_signatures, *assertion_messages],
        token_budget=max(1, self._token_budget // 4),
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
        tests_failed=tests_failed,
        file_line_refs=file_line_refs,
        trimmed_trace=trimmed_trace,
        salient_snippets=salient_snippets,
        stats=stats,
        provenance=Provenance(event_id=event.id),
    )
