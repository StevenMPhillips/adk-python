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

from ..models import Observation
from ..models import PatchCompaction
from ..models import Reflection
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction
from .base_compaction_service import BaseCompactionService


class InMemoryCompactionService(BaseCompactionService):
  """In-memory compaction storage for testing and local development."""

  def __init__(self):
    self._tool_run_compaction_by_event_id: dict[str, ToolRunCompaction] = {}
    self._patch_compaction_by_event_id: dict[str, PatchCompaction] = {}
    self._observations_by_session_id: dict[str, list[Observation]] = {}
    self._reflections_by_session_id: dict[str, list[Reflection]] = {}
    self._task_state_by_session_id: dict[str, TaskStateAnchor] = {}

  async def save_tool_run_compaction(
      self, tool_run_compaction: ToolRunCompaction
  ) -> None:
    self._tool_run_compaction_by_event_id[tool_run_compaction.event_id] = (
        tool_run_compaction
    )

  async def save_patch_compaction(
      self, patch_compaction: PatchCompaction
  ) -> None:
    self._patch_compaction_by_event_id[patch_compaction.event_id] = (
        patch_compaction
    )

  async def save_observation(self, observation: Observation) -> None:
    observations = self._observations_by_session_id.setdefault(
        observation.session_id, []
    )
    observations.append(observation)

  async def save_reflection(self, reflection: Reflection) -> None:
    reflections = self._reflections_by_session_id.setdefault(
        reflection.session_id, []
    )
    reflections.append(reflection)

  async def save_task_state(self, task_state: TaskStateAnchor) -> None:
    self._task_state_by_session_id[task_state.session_id] = task_state

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
