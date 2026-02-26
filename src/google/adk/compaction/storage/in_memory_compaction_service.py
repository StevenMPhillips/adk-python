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

import time

from ..models import Observation
from ..models import PatchCompaction
from ..models import Reflection
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction
from .base_compaction_service import BaseCompactionService
from .base_compaction_service import CompactionCleanupStats


class InMemoryCompactionService(BaseCompactionService):
  """In-memory compaction storage for testing and local development."""

  def __init__(self):
    self._tool_run_compaction_by_event_id: dict[str, ToolRunCompaction] = {}
    self._tool_run_created_meta_by_event_id: dict[str, tuple[float, int]] = {}
    self._patch_compaction_by_event_id: dict[str, PatchCompaction] = {}
    self._patch_created_meta_by_event_id: dict[str, tuple[float, int]] = {}
    self._observations_by_session_id: dict[str, list[Observation]] = {}
    self._observation_created_meta_by_session_id: dict[
        str, list[tuple[float, int]]
    ] = {}
    self._reflections_by_session_id: dict[str, list[Reflection]] = {}
    self._reflection_created_meta_by_session_id: dict[
        str, list[tuple[float, int]]
    ] = {}
    self._task_state_by_session_id: dict[str, TaskStateAnchor] = {}
    self._task_state_updated_meta_by_session_id: dict[
        str, tuple[float, int]
    ] = {}
    self._write_order = 0

  def _next_write_order(self) -> int:
    self._write_order += 1
    return self._write_order

  async def save_tool_run_compaction(
      self, tool_run_compaction: ToolRunCompaction
  ) -> None:
    self._tool_run_compaction_by_event_id[tool_run_compaction.event_id] = (
        tool_run_compaction
    )
    self._tool_run_created_meta_by_event_id[tool_run_compaction.event_id] = (
        time.time(),
        self._next_write_order(),
    )

  async def save_patch_compaction(
      self, patch_compaction: PatchCompaction
  ) -> None:
    self._patch_compaction_by_event_id[patch_compaction.event_id] = (
        patch_compaction
    )
    self._patch_created_meta_by_event_id[patch_compaction.event_id] = (
        time.time(),
        self._next_write_order(),
    )

  async def save_observation(self, observation: Observation) -> None:
    observations = self._observations_by_session_id.setdefault(
        observation.session_id, []
    )
    observations.append(observation)
    observation_meta = self._observation_created_meta_by_session_id.setdefault(
        observation.session_id, []
    )
    observation_meta.append((time.time(), self._next_write_order()))

  async def save_reflection(self, reflection: Reflection) -> None:
    reflections = self._reflections_by_session_id.setdefault(
        reflection.session_id, []
    )
    reflections.append(reflection)
    reflection_meta = self._reflection_created_meta_by_session_id.setdefault(
        reflection.session_id, []
    )
    reflection_meta.append((time.time(), self._next_write_order()))

  async def save_task_state(self, task_state: TaskStateAnchor) -> None:
    self._task_state_by_session_id[task_state.session_id] = task_state
    self._task_state_updated_meta_by_session_id[task_state.session_id] = (
        time.time(),
        self._next_write_order(),
    )

  async def get_tool_run_compaction(
      self, event_id: str
  ) -> ToolRunCompaction | None:
    return self._tool_run_compaction_by_event_id.get(event_id)

  async def get_patch_compaction(self, event_id: str) -> PatchCompaction | None:
    return self._patch_compaction_by_event_id.get(event_id)

  async def get_observations(
      self,
      session_id: str,
      seq_range: tuple[int, int] | None = None,
  ) -> list[Observation]:
    observations = self._observations_by_session_id.get(session_id, [])
    if seq_range is None:
      return list(observations)

    start_seq, end_seq = seq_range
    return [
        observation
        for observation in observations
        if observation.start_seq >= start_seq and observation.end_seq <= end_seq
    ]

  async def get_latest_reflection(self, session_id: str) -> Reflection | None:
    reflections = self._reflections_by_session_id.get(session_id)
    if not reflections:
      return None
    return reflections[-1]

  async def get_task_state(self, session_id: str) -> TaskStateAnchor | None:
    return self._task_state_by_session_id.get(session_id)

  async def query_by_error_signature(
      self, signature: str
  ) -> list[ToolRunCompaction]:
    return [
        tool_run_compaction
        for tool_run_compaction in (
            self._tool_run_compaction_by_event_id.values()
        )
        if signature in tool_run_compaction.error_signatures
    ]

  async def query_by_file_path(
      self, path: str
  ) -> list[ToolRunCompaction | PatchCompaction]:
    matched_artifacts: list[ToolRunCompaction | PatchCompaction] = []

    for tool_run_compaction in self._tool_run_compaction_by_event_id.values():
      if any(ref.path == path for ref in tool_run_compaction.file_line_refs):
        matched_artifacts.append(tool_run_compaction)

    for patch_compaction in self._patch_compaction_by_event_id.values():
      if path in patch_compaction.files_changed:
        matched_artifacts.append(patch_compaction)
        continue
      if any(hunk.path == path for hunk in patch_compaction.hunks):
        matched_artifacts.append(patch_compaction)

    return matched_artifacts

  async def cleanup_artifacts(
      self,
      *,
      max_age_seconds: float | None = None,
      max_records_per_kind: int | None = None,
      session_id: str | None = None,
      now: float | None = None,
  ) -> CompactionCleanupStats:
    if max_age_seconds is None and max_records_per_kind is None:
      return CompactionCleanupStats()

    if max_records_per_kind is not None and max_records_per_kind < 0:
      raise ValueError('max_records_per_kind must be >= 0 when provided.')

    effective_now = time.time() if now is None else now
    cutoff = None
    if max_age_seconds is not None:
      if max_age_seconds < 0:
        raise ValueError('max_age_seconds must be >= 0 when provided.')
      cutoff = effective_now - max_age_seconds

    deleted_tool_runs = 0
    deleted_patches = 0
    deleted_observations = 0
    deleted_reflections = 0
    deleted_task_states = 0

    if session_id is None:
      deleted_tool_runs += self._cleanup_keyed_records(
          record_by_id=self._tool_run_compaction_by_event_id,
          meta_by_id=self._tool_run_created_meta_by_event_id,
          cutoff=cutoff,
          max_records=max_records_per_kind,
      )
      deleted_patches += self._cleanup_keyed_records(
          record_by_id=self._patch_compaction_by_event_id,
          meta_by_id=self._patch_created_meta_by_event_id,
          cutoff=cutoff,
          max_records=max_records_per_kind,
      )

    observation_sessions = (
        [session_id]
        if session_id is not None
        else list(self._observations_by_session_id.keys())
    )
    for observation_session_id in observation_sessions:
      deleted_observations += self._cleanup_session_lists(
          session_id=observation_session_id,
          records_by_session_id=self._observations_by_session_id,
          meta_by_session_id=self._observation_created_meta_by_session_id,
          cutoff=cutoff,
          max_records=max_records_per_kind,
      )

    reflection_sessions = (
        [session_id]
        if session_id is not None
        else list(self._reflections_by_session_id.keys())
    )
    for reflection_session_id in reflection_sessions:
      deleted_reflections += self._cleanup_session_lists(
          session_id=reflection_session_id,
          records_by_session_id=self._reflections_by_session_id,
          meta_by_session_id=self._reflection_created_meta_by_session_id,
          cutoff=cutoff,
          max_records=max_records_per_kind,
      )

    deleted_task_states += self._cleanup_task_states(
        cutoff=cutoff,
        max_records=max_records_per_kind,
        session_id=session_id,
    )

    return CompactionCleanupStats(
        tool_run_compactions_deleted=deleted_tool_runs,
        patch_compactions_deleted=deleted_patches,
        observations_deleted=deleted_observations,
        reflections_deleted=deleted_reflections,
        task_states_deleted=deleted_task_states,
    )

  async def evict_session(self, session_id: str) -> CompactionCleanupStats:
    observations_deleted = len(
        self._observations_by_session_id.get(session_id, [])
    )
    reflections_deleted = len(
        self._reflections_by_session_id.get(session_id, [])
    )
    task_states_deleted = (
        1 if session_id in self._task_state_by_session_id else 0
    )

    self._observations_by_session_id.pop(session_id, None)
    self._observation_created_meta_by_session_id.pop(session_id, None)
    self._reflections_by_session_id.pop(session_id, None)
    self._reflection_created_meta_by_session_id.pop(session_id, None)
    self._task_state_by_session_id.pop(session_id, None)
    self._task_state_updated_meta_by_session_id.pop(session_id, None)

    return CompactionCleanupStats(
        observations_deleted=observations_deleted,
        reflections_deleted=reflections_deleted,
        task_states_deleted=task_states_deleted,
    )

  def _cleanup_keyed_records(
      self,
      *,
      record_by_id: dict[str, object],
      meta_by_id: dict[str, tuple[float, int]],
      cutoff: float | None,
      max_records: int | None,
  ) -> int:
    deleted = 0

    if cutoff is not None:
      stale_ids = [
          record_id
          for record_id, (created_at, _) in meta_by_id.items()
          if created_at <= cutoff
      ]
      for stale_id in stale_ids:
        if stale_id in record_by_id:
          del record_by_id[stale_id]
          deleted += 1
        meta_by_id.pop(stale_id, None)

    if max_records is not None and len(record_by_id) > max_records:
      sorted_ids = sorted(
          record_by_id.keys(),
          key=lambda record_id: (*meta_by_id[record_id], record_id),
      )
      removable_ids = sorted_ids[: len(record_by_id) - max_records]
      for removable_id in removable_ids:
        del record_by_id[removable_id]
        meta_by_id.pop(removable_id, None)
      deleted += len(removable_ids)

    return deleted

  def _cleanup_session_lists(
      self,
      *,
      session_id: str,
      records_by_session_id: dict[str, list[Observation] | list[Reflection]],
      meta_by_session_id: dict[str, list[tuple[float, int]]],
      cutoff: float | None,
      max_records: int | None,
  ) -> int:
    records = records_by_session_id.get(session_id)
    metadata = meta_by_session_id.get(session_id)
    if not records or not metadata:
      return 0

    survivors = [
        (record, meta)
        for record, meta in zip(records, metadata)
        if cutoff is None or meta[0] > cutoff
    ]
    deleted = len(records) - len(survivors)

    if max_records is not None and len(survivors) > max_records:
      sorted_indices = sorted(
          range(len(survivors)), key=lambda idx: (*survivors[idx][1], idx)
      )
      remove_count = len(survivors) - max_records
      drop_indices = set(sorted_indices[:remove_count])
      survivors = [
          survivor
          for idx, survivor in enumerate(survivors)
          if idx not in drop_indices
      ]
      deleted += remove_count

    if not survivors:
      records_by_session_id.pop(session_id, None)
      meta_by_session_id.pop(session_id, None)
      return deleted

    records_by_session_id[session_id] = [record for record, _ in survivors]
    meta_by_session_id[session_id] = [meta for _, meta in survivors]
    return deleted

  def _cleanup_task_states(
      self,
      *,
      cutoff: float | None,
      max_records: int | None,
      session_id: str | None,
  ) -> int:
    deleted = 0
    session_ids = (
        [session_id]
        if session_id is not None
        else list(self._task_state_by_session_id.keys())
    )

    if cutoff is not None:
      stale_session_ids = [
          candidate_session_id
          for candidate_session_id in session_ids
          if self._task_state_updated_meta_by_session_id.get(
              candidate_session_id, (0.0, 0)
          )[0]
          <= cutoff
      ]
      for stale_session_id in stale_session_ids:
        if stale_session_id in self._task_state_by_session_id:
          del self._task_state_by_session_id[stale_session_id]
          deleted += 1
        self._task_state_updated_meta_by_session_id.pop(stale_session_id, None)

    if max_records is not None and session_id is None:
      record_count = len(self._task_state_by_session_id)
      if record_count > max_records:
        sorted_session_ids = sorted(
            self._task_state_by_session_id.keys(),
            key=lambda candidate_session_id: (
                *self._task_state_updated_meta_by_session_id[
                    candidate_session_id
                ],
                candidate_session_id,
            ),
        )
        stale_session_ids = sorted_session_ids[: record_count - max_records]
        for stale_session_id in stale_session_ids:
          del self._task_state_by_session_id[stale_session_id]
          self._task_state_updated_meta_by_session_id.pop(
              stale_session_id, None
          )
        deleted += len(stale_session_ids)

    return deleted
