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

import dataclasses
import re

from ...events.event import Event
from ..models import CompactionStats
from ..models import HunkSummary
from ..models import PatchCompaction
from ..models import Provenance
from .generic import _estimate_token_count
from .generic import _extract_response_payload
from .generic import _safe_to_text

_DEFAULT_TOKEN_BUDGET = 800
_MAX_SNIPPET_LINES = 8
_MAX_SNIPPET_CHARS = 480
_HUNK_HEADER_PATTERN = re.compile(
    r'^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+'
    r'\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@'
)


@dataclasses.dataclass(frozen=True)
class _ParsedHunk:
  """Internal representation of one parsed unified-diff hunk."""

  path: str
  change_size: int
  anchor_before: str
  anchor_after: str
  snippet: str


def _normalize_diff_path(raw_path: str) -> str:
  """Normalizes a unified diff file marker path."""
  path = raw_path.strip().split('\t', 1)[0]
  if path in ('/dev/null', 'null'):
    return ''
  if path.startswith('a/') or path.startswith('b/'):
    return path[2:]
  return path


def _extract_anchor(lines: list[str], prefixes: tuple[str, ...]) -> str:
  """Returns the first non-empty text from matching hunk lines."""
  for line in lines:
    if not line or line.startswith('\\'):
      continue
    if line[0] not in prefixes:
      continue
    text = line[1:].strip()
    if text:
      return text
  return ''


def _build_trimmed_snippet(header: str, lines: list[str]) -> str:
  """Builds a compact, readable snippet for a parsed hunk."""
  snippet_lines = [header, *lines]
  if len(snippet_lines) > _MAX_SNIPPET_LINES:
    snippet_lines = snippet_lines[: _MAX_SNIPPET_LINES - 1] + ['...']

  trimmed_lines: list[str] = []
  char_budget = _MAX_SNIPPET_CHARS
  for line in snippet_lines:
    line_chars = len(line) + 1
    if line_chars > char_budget:
      break
    trimmed_lines.append(line)
    char_budget -= line_chars

  if not trimmed_lines:
    return header
  return '\n'.join(trimmed_lines)


def _extract_patch_text(payload: dict[str, object]) -> str:
  """Extracts patch text from tool payloads."""
  for key in ('diff', 'patch', 'stdout'):
    value = payload.get(key)
    text = _safe_to_text(value)
    if text.strip():
      return text
  return ''


def _parse_unified_diff(
    patch_text: str,
) -> tuple[list[str], dict[str, int], list[_ParsedHunk], set[str]]:
  """Parses changed files, per-file hunk counts, and hunk summaries."""
  files_changed: list[str] = []
  seen_files: set[str] = set()
  hunk_count_by_file: dict[str, int] = {}
  parsed_hunks: list[_ParsedHunk] = []
  operation_tags: set[str] = set()

  current_path = ''
  current_hunk_header = ''
  current_hunk_lines: list[str] | None = None
  old_marker = ''

  def finalize_hunk() -> None:
    nonlocal current_hunk_header
    nonlocal current_hunk_lines
    if (
        current_hunk_lines is None
        or not current_path
        or not current_hunk_header
    ):
      current_hunk_header = ''
      current_hunk_lines = None
      return
    change_size = sum(
        1
        for line in current_hunk_lines
        if line.startswith('+') or line.startswith('-')
    )
    parsed_hunks.append(
        _ParsedHunk(
            path=current_path,
            change_size=change_size,
            anchor_before=_extract_anchor(current_hunk_lines, (' ', '-')),
            anchor_after=_extract_anchor(current_hunk_lines, (' ', '+')),
            snippet=_build_trimmed_snippet(
                current_hunk_header,
                current_hunk_lines,
            ),
        )
    )
    hunk_count_by_file[current_path] = (
        hunk_count_by_file.get(current_path, 0) + 1
    )
    current_hunk_header = ''
    current_hunk_lines = None

  for line in patch_text.splitlines():
    if line.startswith('--- '):
      finalize_hunk()
      old_marker = _normalize_diff_path(line[4:])
      continue

    if line.startswith('+++ '):
      finalize_hunk()
      new_marker = _normalize_diff_path(line[4:])
      if old_marker and new_marker:
        operation_tags.add('modified')
      elif new_marker:
        operation_tags.add('added')
      elif old_marker:
        operation_tags.add('deleted')
      current_path = new_marker or old_marker
      if current_path and current_path not in seen_files:
        seen_files.add(current_path)
        files_changed.append(current_path)
      continue

    if line.startswith('@@ '):
      if not _HUNK_HEADER_PATTERN.match(line):
        continue
      finalize_hunk()
      current_hunk_header = line
      current_hunk_lines = []
      continue

    if current_hunk_lines is not None:
      if line.startswith('diff --git '):
        finalize_hunk()
        current_path = ''
        continue
      if line.startswith((' ', '+', '-', '\\')):
        current_hunk_lines.append(line)

  finalize_hunk()
  return files_changed, hunk_count_by_file, parsed_hunks, operation_tags


def _select_hunks_with_budget(
    parsed_hunks: list[_ParsedHunk],
    *,
    token_budget: int,
    files_changed: list[str],
    semantic_tags: list[str],
) -> list[HunkSummary]:
  """Selects hunks, prioritizing larger ones when budget is tight."""
  if not parsed_hunks or token_budget <= 0:
    return []

  baseline_tokens = _estimate_token_count(
      '\n'.join([*files_changed, *semantic_tags])
  )
  remaining_budget = max(0, token_budget - baseline_tokens)

  selected: list[_ParsedHunk] = []
  consumed_tokens = 0

  for hunk in sorted(
      parsed_hunks, key=lambda item: item.change_size, reverse=True
  ):
    hunk_tokens = _estimate_token_count(
        '\n'.join(
            [hunk.path, hunk.anchor_before, hunk.anchor_after, hunk.snippet]
        )
    )
    if consumed_tokens + hunk_tokens > remaining_budget:
      if not selected:
        selected = [hunk]
        break
      continue
    selected.append(hunk)
    consumed_tokens += hunk_tokens

  if not selected:
    largest_hunk = max(parsed_hunks, key=lambda item: item.change_size)
    selected = [largest_hunk]

  return [
      HunkSummary(
          path=hunk.path,
          anchor_before=hunk.anchor_before,
          anchor_after=hunk.anchor_after,
          snippet=hunk.snippet,
      )
      for hunk in selected
  ]


def _build_compact_text(
    *,
    files_changed: list[str],
    semantic_tags: list[str],
    hunks: list[HunkSummary],
) -> str:
  """Builds canonical compacted text used for token estimation."""
  return '\n'.join([
      *files_changed,
      *semantic_tags,
      *[
          '\n'.join([
              hunk.path,
              hunk.anchor_before,
              hunk.anchor_after,
              hunk.snippet,
          ])
          for hunk in hunks
      ],
  ])


class PatchCompactor:
  """Compactor specialized for unified diff patch output."""

  def __init__(self, *, token_budget: int = _DEFAULT_TOKEN_BUDGET):
    self._token_budget = max(0, token_budget)

  def compact(self, event: Event) -> PatchCompaction | None:
    _, payload = _extract_response_payload(event)
    if not payload:
      return None

    patch_text = _extract_patch_text(payload)
    if not patch_text:
      return None

    files_changed, hunk_count_by_file, parsed_hunks, operation_tags = (
        _parse_unified_diff(patch_text)
    )
    if not files_changed or not parsed_hunks:
      return None

    semantic_tags = ['unified-diff']
    semantic_tags.extend(sorted(operation_tags))
    if len(files_changed) > 1:
      semantic_tags.append('multi-file')
    else:
      semantic_tags.append('single-file')
    optional_hunk_tags = sorted(
        f'hunk-count:{path}={count}'
        for path, count in hunk_count_by_file.items()
    )
    semantic_tags.extend(optional_hunk_tags)

    hunks = _select_hunks_with_budget(
        parsed_hunks,
        token_budget=self._token_budget,
        files_changed=files_changed,
        semantic_tags=semantic_tags,
    )

    compact_text = _build_compact_text(
        files_changed=files_changed,
        semantic_tags=semantic_tags,
        hunks=hunks,
    )
    if (
        self._token_budget > 0
        and _estimate_token_count(compact_text) > self._token_budget
    ):
      while optional_hunk_tags:
        semantic_tags.remove(optional_hunk_tags.pop())
        compact_text = _build_compact_text(
            files_changed=files_changed,
            semantic_tags=semantic_tags,
            hunks=hunks,
        )
        if _estimate_token_count(compact_text) <= self._token_budget:
          break

    if self._token_budget > 0 and hunks:
      while (
          len(hunks) > 1
          and _estimate_token_count(compact_text) > self._token_budget
      ):
        hunks = hunks[:-1]
        compact_text = _build_compact_text(
            files_changed=files_changed,
            semantic_tags=semantic_tags,
            hunks=hunks,
        )
    raw_tokens_est = _estimate_token_count(patch_text)
    compact_tokens_est = _estimate_token_count(compact_text)
    if compact_tokens_est <= 0:
      compression_ratio = float(raw_tokens_est) if raw_tokens_est else 1.0
    else:
      compression_ratio = raw_tokens_est / compact_tokens_est

    return PatchCompaction(
        event_id=event.id,
        compaction_version=1,
        files_changed=files_changed,
        hunks=hunks,
        semantic_tags=semantic_tags,
        stats=CompactionStats(
            raw_tokens_est=raw_tokens_est,
            compact_tokens_est=compact_tokens_est,
            compression_ratio=compression_ratio,
        ),
        provenance=Provenance(event_id=event.id),
    )
