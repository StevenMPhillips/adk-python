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
from typing import Any

from ...events.event import Event
from ..models import FileLineRef
from ..models import Provenance
from ..models import ToolRunCompaction
from .base import BaseToolRunCompactor
from .helpers import _build_compaction_stats
from .helpers import _build_raw_text

_DEFAULT_STDERR_TAIL_LINES = 20
_DEFAULT_TOKEN_BUDGET = 600
_CHARS_PER_TOKEN = 4

_PATH_SEGMENT = r'[A-Za-z0-9._@+-]+'
_RELATIVE_OR_UNIX_PATH = (
    rf'(?:{_PATH_SEGMENT}(?:[\\/]{_PATH_SEGMENT})*|'
    rf'(?:\.|\.\.|~)(?:[\\/]{_PATH_SEGMENT})+|'
    rf'[\\/]{_PATH_SEGMENT}(?:[\\/]{_PATH_SEGMENT})*)'
)
_WINDOWS_DRIVE_PATH = rf'[A-Za-z]:(?:[\\/]{_PATH_SEGMENT})+'
_WINDOWS_UNC_PATH = rf'\\\\{_PATH_SEGMENT}(?:\\{_PATH_SEGMENT})+'
_FILE_LINE_PATTERN = re.compile(
    rf'(?<![A-Za-z0-9_.])'
    r'(?=[^:\n]*[A-Za-z_./\\~@-])'
    rf'(?P<path>(?:{_WINDOWS_DRIVE_PATH}|{_WINDOWS_UNC_PATH}|'
    rf'{_RELATIVE_OR_UNIX_PATH}))'
    r':(?P<line>\d+)(?::(?P<col>\d+))?'
    r'(?=$|[^0-9])'
)


def _estimate_token_count(text: str) -> int:
  """Returns a rough token estimate with a 4 chars/token heuristic."""
  return len(text) // _CHARS_PER_TOKEN


def _safe_to_int(value: Any) -> int | None:
  """Converts a dynamic value to int when possible."""
  if isinstance(value, bool):
    return None
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(value)
  if isinstance(value, str):
    stripped = value.strip()
    if not stripped:
      return None
    try:
      return int(stripped)
    except ValueError:
      return None
  return None


def _safe_to_text(value: Any) -> str:
  """Converts unknown values to displayable text."""
  if value is None:
    return ''
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    return '\n'.join(_safe_to_text(item) for item in value)
  if isinstance(value, dict):
    return json.dumps(value, sort_keys=True)
  return str(value)


def _extract_response_payload(event: Event) -> tuple[str, dict[str, Any]]:
  """Extracts a function response payload and function name."""
  for function_response in event.get_function_responses():
    if not function_response:
      continue
    response = function_response.response
    if isinstance(response, dict):
      return function_response.name or '', response
  return '', {}


def _extract_exit_code(payload: dict[str, Any]) -> int:
  """Extracts exit code-like fields from a response payload."""
  for key in ('exit_code', 'exitCode', 'returncode', 'return_code', 'code'):
    exit_code = _safe_to_int(payload.get(key))
    if exit_code is not None:
      return exit_code
  return 0


def _extract_command(function_name: str, payload: dict[str, Any]) -> str:
  """Extracts command text for registry matching and diagnostics."""
  for key in ('command', 'cmd'):
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()

  tool_input = payload.get('tool_input')
  if isinstance(tool_input, dict):
    command = tool_input.get('command')
    if isinstance(command, str) and command.strip():
      return command.strip()

  return function_name


def _cap_lines_by_token_budget(
    lines: list[str], *, token_budget: int
) -> list[str]:
  """Caps lines to a token budget while preserving the latest lines."""
  if token_budget <= 0:
    return []

  max_chars = token_budget * _CHARS_PER_TOKEN
  total_chars = 0
  kept_reversed: list[str] = []

  for line in reversed(lines):
    line_chars = len(line) + 1
    if total_chars + line_chars > max_chars:
      break
    kept_reversed.append(line)
    total_chars += line_chars

  kept_reversed.reverse()
  return kept_reversed


def _extract_file_line_refs(text: str) -> list[FileLineRef]:
  """Extracts unique file:line(:col) references from text."""
  refs: list[FileLineRef] = []
  seen: set[tuple[str, int, int | None]] = set()

  for match in _FILE_LINE_PATTERN.finditer(text):
    path = match.group('path')
    line = int(match.group('line'))
    col_match = match.group('col')
    col = int(col_match) if col_match is not None else None
    key = (path, line, col)
    if key in seen:
      continue
    seen.add(key)
    refs.append(FileLineRef(path=path, line=line, col=col))

  return refs


class GenericToolRunCompactor(BaseToolRunCompactor):
  """Fallback compactor for generic tool output streams."""

  def __init__(
      self,
      *,
      stderr_tail_lines: int = _DEFAULT_STDERR_TAIL_LINES,
      token_budget: int = _DEFAULT_TOKEN_BUDGET,
  ):
    self._stderr_tail_lines = max(0, stderr_tail_lines)
    self._token_budget = max(0, token_budget)

  def compact(self, event: Event) -> ToolRunCompaction | None:
    function_name, payload = _extract_response_payload(event)
    if not payload:
      return None

    command = _extract_command(function_name, payload)
    exit_code = _extract_exit_code(payload)

    stderr_text = _safe_to_text(payload.get('stderr'))
    stdout_text = _safe_to_text(payload.get('stdout'))
    stderr_lines = stderr_text.splitlines()
    tail_lines = stderr_lines[-self._stderr_tail_lines :]
    trimmed_trace = _cap_lines_by_token_budget(
        tail_lines,
        token_budget=self._token_budget,
    )

    refs = _extract_file_line_refs('\n'.join([stdout_text, stderr_text]))

    key_errors = [line for line in trimmed_trace if line.strip()][:3]

    compact_text = '\n'.join(trimmed_trace)
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
        error_signatures=[],
        key_errors=key_errors,
        tests_failed=[],
        file_line_refs=refs,
        trimmed_trace=trimmed_trace,
        salient_snippets=[],
        stats=stats,
        provenance=Provenance(event_id=event.id),
    )
