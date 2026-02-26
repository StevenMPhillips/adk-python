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

from collections import Counter
from collections.abc import Mapping
import re
from typing import Literal

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict

from .models import Observation
from .models import PatchCompaction
from .models import TaskStateAnchor
from .models import ToolRunCompaction
from .storage.base_compaction_service import BaseCompactionService

CompactionArtifact = ToolRunCompaction | PatchCompaction

_FILE_PATH_RE = re.compile(
    r'(?P<path>[A-Za-z0-9_./-]+\.(?:py|pyi|md|txt|json|yaml|yml))'
)
_TEST_NAME_RE = re.compile(r'(?P<test>[A-Za-z0-9_./-]+::[A-Za-z0-9_\[\].-]+)')
_SIGNATURE_RE = re.compile(r'(?P<sig>[A-Z][A-Za-z0-9_]+(?:Error|Exception))')
_OLDER_REFERENCE_RE = re.compile(
    r'\b(earlier|before|previous|prior|last\s+time|older)\b',
    re.IGNORECASE,
)


class _RehydrationModel(BaseModel):
  """Shared base model for rehydration planner and executor models."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )


class TimeRange(_RehydrationModel):
  """Inclusive range used for sequence or timestamp constrained retrieval."""

  start: int | None = None
  end: int | None = None


class RetrievalQuery(_RehydrationModel):
  """Structured retrieval request produced by the rehydration planner."""

  trigger: Literal[
      'missing_evidence',
      'thrash_detection',
      'user_references_older_specifics',
  ]
  reason: str
  error_signature: str | None = None
  test_name: str | None = None
  file_path: str | None = None
  time_range: TimeRange | None = None


class RawExcerpt(_RehydrationModel):
  """Raw text excerpt associated with an artifact event id."""

  event_id: str
  excerpt: str


class EvidencePack(_RehydrationModel):
  """Hydrated evidence payload used by hybrid prompt assembly."""

  token_budget: int = 2000
  tokens_est: int = 0
  raw_excerpts: list[RawExcerpt]
  compacted_artifacts: list[CompactionArtifact]


class RehydrationPlanner:
  """Builds retrieval queries from task state, memory, and user intent."""

  def __init__(self, thrash_repeat_threshold: int = 2):
    self._thrash_repeat_threshold = max(2, thrash_repeat_threshold)

  def plan(
      self,
      *,
      task_state: TaskStateAnchor,
      recent_observations: list[Observation],
      recent_compactions: list[CompactionArtifact],
      latest_user_message: str,
  ) -> list[RetrievalQuery]:
    """Generates retrieval queries for missing evidence and thrash loops."""

    time_range = self._time_range_from_observations(recent_observations)
    queries: list[RetrievalQuery] = []

    missing_query = self._missing_evidence_query(task_state, time_range)
    if missing_query is not None:
      queries.append(missing_query)

    thrash_query = self._thrash_query(recent_compactions, time_range)
    if thrash_query is not None:
      queries.append(thrash_query)

    user_query = self._older_user_reference_query(
        latest_user_message=latest_user_message,
        recent_compactions=recent_compactions,
        time_range=time_range,
    )
    if user_query is not None:
      queries.append(user_query)

    return self._dedupe_queries(queries)

  def _time_range_from_observations(
      self, recent_observations: list[Observation]
  ) -> TimeRange | None:
    if not recent_observations:
      return None
    return TimeRange(
        start=min(obs.start_seq for obs in recent_observations),
        end=max(obs.end_seq for obs in recent_observations),
    )

  def _missing_evidence_query(
      self,
      task_state: TaskStateAnchor,
      time_range: TimeRange | None,
  ) -> RetrievalQuery | None:
    for item in (
        task_state.constraints
        + task_state.hypotheses
        + task_state.known_failures
        + task_state.current_plan
        + task_state.next_steps
    ):
      if item.evidence_refs:
        continue
      signature = _first_match(_SIGNATURE_RE, item.text, 'sig')
      test_name = _first_match(_TEST_NAME_RE, item.text, 'test')
      file_path = _best_file_path(item.text)
      return RetrievalQuery(
          trigger='missing_evidence',
          reason='Task state contains entries without evidence refs.',
          error_signature=signature,
          test_name=test_name,
          file_path=file_path,
          time_range=time_range,
      )
    return None

  def _thrash_query(
      self,
      recent_compactions: list[CompactionArtifact],
      time_range: TimeRange | None,
  ) -> RetrievalQuery | None:
    signatures = [
        signature
        for artifact in recent_compactions
        if isinstance(artifact, ToolRunCompaction)
        for signature in artifact.error_signatures
    ]
    if not signatures:
      return None

    most_common_signature, count = Counter(signatures).most_common(1)[0]
    if count < self._thrash_repeat_threshold:
      return None

    test_name = None
    for artifact in recent_compactions:
      if not isinstance(artifact, ToolRunCompaction):
        continue
      if most_common_signature in artifact.error_signatures:
        if artifact.tests_failed:
          test_name = artifact.tests_failed[0]
          break

    return RetrievalQuery(
        trigger='thrash_detection',
        reason='Repeated error signatures detected in recent attempts.',
        error_signature=most_common_signature,
        test_name=test_name,
        time_range=time_range,
    )

  def _older_user_reference_query(
      self,
      *,
      latest_user_message: str,
      recent_compactions: list[CompactionArtifact],
      time_range: TimeRange | None,
  ) -> RetrievalQuery | None:
    if not _OLDER_REFERENCE_RE.search(latest_user_message):
      return None

    file_path = _best_file_path(latest_user_message)
    test_name = _first_match(_TEST_NAME_RE, latest_user_message, 'test')
    signature = _first_match(_SIGNATURE_RE, latest_user_message, 'sig')

    if signature is None:
      for artifact in recent_compactions:
        if not isinstance(artifact, ToolRunCompaction):
          continue
        if artifact.error_signatures:
          signature = artifact.error_signatures[0]
          break

    return RetrievalQuery(
        trigger='user_references_older_specifics',
        reason='User asked to revisit older specifics.',
        error_signature=signature,
        test_name=test_name,
        file_path=file_path,
        time_range=time_range,
    )

  def _dedupe_queries(
      self, queries: list[RetrievalQuery]
  ) -> list[RetrievalQuery]:
    deduped: list[RetrievalQuery] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    for query in queries:
      key = (
          query.trigger,
          query.error_signature,
          query.test_name,
          query.file_path,
      )
      if key in seen:
        continue
      seen.add(key)
      deduped.append(query)
    return deduped


class RehydrationExecutor:
  """Executes retrieval queries against compaction storage."""

  def __init__(
      self,
      compaction_service: BaseCompactionService,
      *,
      token_budget: int = 2000,
  ):
    self._compaction_service = compaction_service
    self._token_budget = token_budget

  async def execute(
      self,
      queries: list[RetrievalQuery],
      *,
      event_time_by_id: Mapping[str, int] | None = None,
      raw_excerpt_by_event_id: Mapping[str, str] | None = None,
  ) -> EvidencePack:
    """Fetches and packs compacted artifacts and excerpts into a budget."""

    matched_artifacts: list[CompactionArtifact] = []
    seen_event_ids: set[str] = set()

    for query in queries:
      artifacts = await self._retrieve_for_query(
          query=query,
          event_time_by_event_id=event_time_by_id,
      )
      for artifact in artifacts:
        if artifact.event_id in seen_event_ids:
          continue
        seen_event_ids.add(artifact.event_id)
        matched_artifacts.append(artifact)

    return self._build_evidence_pack(
        matched_artifacts=matched_artifacts,
        raw_excerpt_by_event_id=raw_excerpt_by_event_id,
    )

  async def _retrieve_for_query(
      self,
      *,
      query: RetrievalQuery,
      event_time_by_event_id: Mapping[str, int] | None,
  ) -> list[CompactionArtifact]:
    artifacts: list[CompactionArtifact] = []

    if query.error_signature:
      artifacts.extend(
          await self._compaction_service.query_by_error_signature(
              query.error_signature
          )
      )

    if query.file_path:
      artifacts.extend(
          await self._compaction_service.query_by_file_path(query.file_path)
      )

    if not query.error_signature and not query.file_path:
      return []

    filtered = self._apply_query_filters(
        artifacts=artifacts,
        query=query,
        event_time_by_event_id=event_time_by_event_id,
    )
    return filtered

  def _apply_query_filters(
      self,
      *,
      artifacts: list[CompactionArtifact],
      query: RetrievalQuery,
      event_time_by_event_id: Mapping[str, int] | None,
  ) -> list[CompactionArtifact]:
    filtered = list(artifacts)

    if query.error_signature:
      filtered = [
          artifact
          for artifact in filtered
          if not isinstance(artifact, ToolRunCompaction)
          or query.error_signature in artifact.error_signatures
      ]

    if query.test_name:
      filtered = [
          artifact
          for artifact in filtered
          if not isinstance(artifact, ToolRunCompaction)
          or any(query.test_name in test for test in artifact.tests_failed)
      ]

    if query.file_path:
      filtered = [
          artifact
          for artifact in filtered
          if _artifact_mentions_file(artifact, query.file_path)
      ]

    if query.time_range and event_time_by_event_id:
      filtered = [
          artifact
          for artifact in filtered
          if _in_time_range(
              event_time_by_event_id.get(artifact.event_id), query.time_range
          )
      ]

    deduped: list[CompactionArtifact] = []
    seen = set()
    for artifact in filtered:
      if artifact.event_id in seen:
        continue
      seen.add(artifact.event_id)
      deduped.append(artifact)
    return deduped

  def _build_evidence_pack(
      self,
      *,
      matched_artifacts: list[CompactionArtifact],
      raw_excerpt_by_event_id: Mapping[str, str] | None,
  ) -> EvidencePack:
    used_tokens = 0
    included_artifacts: list[CompactionArtifact] = []
    raw_excerpts: list[RawExcerpt] = []

    for artifact in matched_artifacts:
      artifact_tokens = _artifact_tokens(artifact)
      excerpt_text = _resolve_excerpt(artifact, raw_excerpt_by_event_id)
      excerpt_tokens = _estimate_tokens(excerpt_text)
      projected_tokens = used_tokens + artifact_tokens + excerpt_tokens
      if projected_tokens > self._token_budget:
        continue

      included_artifacts.append(artifact)
      raw_excerpts.append(
          RawExcerpt(event_id=artifact.event_id, excerpt=excerpt_text)
      )
      used_tokens = projected_tokens

    return EvidencePack(
        token_budget=self._token_budget,
        tokens_est=used_tokens,
        raw_excerpts=raw_excerpts,
        compacted_artifacts=included_artifacts,
    )


def _first_match(pattern: re.Pattern[str], text: str, group: str) -> str | None:
  match = pattern.search(text)
  if not match:
    return None
  return match.group(group)


def _best_file_path(text: str) -> str | None:
  matches = [match.group('path') for match in _FILE_PATH_RE.finditer(text)]
  if not matches:
    return None
  for path in matches:
    if path.startswith('src/'):
      return path
  return matches[0]


def _artifact_mentions_file(artifact: CompactionArtifact, path: str) -> bool:
  if isinstance(artifact, ToolRunCompaction):
    return any(ref.path == path for ref in artifact.file_line_refs)
  return path in artifact.files_changed or any(
      hunk.path == path for hunk in artifact.hunks
  )


def _in_time_range(value: int | None, time_range: TimeRange) -> bool:
  if value is None:
    return False
  if time_range.start is not None and value < time_range.start:
    return False
  if time_range.end is not None and value > time_range.end:
    return False
  return True


def _artifact_tokens(artifact: CompactionArtifact) -> int:
  return artifact.stats.compact_tokens_est


def _resolve_excerpt(
    artifact: CompactionArtifact,
    raw_excerpt_by_event_id: Mapping[str, str] | None,
) -> str:
  if raw_excerpt_by_event_id and artifact.event_id in raw_excerpt_by_event_id:
    return raw_excerpt_by_event_id[artifact.event_id]

  if isinstance(artifact, ToolRunCompaction):
    if artifact.salient_snippets:
      return artifact.salient_snippets[0]
    if artifact.trimmed_trace:
      return artifact.trimmed_trace[0]
    return artifact.command

  if artifact.hunks:
    return artifact.hunks[0].snippet
  if artifact.files_changed:
    return artifact.files_changed[0]
  return artifact.event_id


def _estimate_tokens(text: str) -> int:
  return max(1, len(text) // 4)
