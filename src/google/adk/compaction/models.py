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

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict


class _CompactionModel(BaseModel):
  """Shared base model for compaction artifacts."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )


class CompactionStats(_CompactionModel):
  """Token counts and ratio for compaction output."""

  raw_tokens_est: int
  compact_tokens_est: int
  compression_ratio: float


class FileLineRef(_CompactionModel):
  """Reference to a specific source file location."""

  path: str
  line: int
  col: int | None = None


class HunkSummary(_CompactionModel):
  """Compact summary of a changed patch hunk."""

  path: str
  anchor_before: str
  anchor_after: str
  snippet: str


class Provenance(_CompactionModel):
  """Traceability metadata to the originating event streams."""

  event_id: str
  stdout_offsets: tuple[int, int] | None = None
  stderr_offsets: tuple[int, int] | None = None


class ToolRunCompaction(_CompactionModel):
  """Compacted artifact describing a tool-run execution event."""

  event_id: str
  compaction_version: int
  command: str
  exit_code: int
  duration_ms: int | None = None
  error_signatures: list[str]
  key_errors: list[str]
  tests_failed: list[str]
  file_line_refs: list[FileLineRef]
  trimmed_trace: list[str]
  salient_snippets: list[str]
  stats: CompactionStats
  provenance: Provenance


class PatchCompaction(_CompactionModel):
  """Compacted artifact describing a patch/diff event."""

  event_id: str
  compaction_version: int
  files_changed: list[str]
  hunks: list[HunkSummary]
  semantic_tags: list[str]
  stats: CompactionStats
  provenance: Provenance
