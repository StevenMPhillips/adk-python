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

import abc

from ..models import Observation
from ..models import PatchCompaction
from ..models import Reflection
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction


class BaseCompactionService(abc.ABC):
  """Base interface for compaction artifact persistence."""

  @abc.abstractmethod
  async def save_tool_run_compaction(
      self, tool_run_compaction: ToolRunCompaction
  ) -> None:
    """Persists a compacted tool-run artifact."""

  @abc.abstractmethod
  async def save_patch_compaction(
      self, patch_compaction: PatchCompaction
  ) -> None:
    """Persists a compacted patch artifact."""

  @abc.abstractmethod
  async def save_observation(self, observation: Observation) -> None:
    """Persists an observation artifact."""

  @abc.abstractmethod
  async def save_reflection(self, reflection: Reflection) -> None:
    """Persists a reflection artifact."""

  @abc.abstractmethod
  async def save_task_state(self, task_state: TaskStateAnchor) -> None:
    """Persists the latest task-state snapshot for a session."""

  @abc.abstractmethod
  async def get_tool_run_compaction(
      self, event_id: str
  ) -> ToolRunCompaction | None:
    """Returns a compacted tool-run artifact by source event id."""

  @abc.abstractmethod
  async def get_patch_compaction(self, event_id: str) -> PatchCompaction | None:
    """Returns a compacted patch artifact by source event id."""

  @abc.abstractmethod
  async def get_observations(
      self,
      session_id: str,
      seq_range: tuple[int, int] | None = None,
  ) -> list[Observation]:
    """Returns observations for a session, optionally filtered by range."""

  @abc.abstractmethod
  async def get_latest_reflection(self, session_id: str) -> Reflection | None:
    """Returns the latest reflection recorded for the session."""

  @abc.abstractmethod
  async def get_task_state(self, session_id: str) -> TaskStateAnchor | None:
    """Returns the latest task-state snapshot for the session."""

  @abc.abstractmethod
  async def query_by_error_signature(
      self, signature: str
  ) -> list[ToolRunCompaction]:
    """Returns tool-run compact artifacts matching an error signature."""

  @abc.abstractmethod
  async def query_by_file_path(
      self, path: str
  ) -> list[ToolRunCompaction | PatchCompaction]:
    """Returns compaction artifacts that reference a file path."""
