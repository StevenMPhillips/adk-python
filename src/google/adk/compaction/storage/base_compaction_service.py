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
from dataclasses import dataclass

from ..models import Observation
from ..models import PatchCompaction
from ..models import Reflection
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction


@dataclass(frozen=True)
class CompactionCleanupStats:
  """Counts of artifacts removed by a cleanup operation."""

  tool_run_compactions_deleted: int = 0
  patch_compactions_deleted: int = 0
  observations_deleted: int = 0
  reflections_deleted: int = 0
  task_states_deleted: int = 0

  @property
  def total_deleted(self) -> int:
    """Returns the total number of deleted records across all kinds."""
    return (
        self.tool_run_compactions_deleted
        + self.patch_compactions_deleted
        + self.observations_deleted
        + self.reflections_deleted
        + self.task_states_deleted
    )


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

  async def cleanup_artifacts(
      self,
      *,
      max_age_seconds: float | None = None,
      max_records_per_kind: int | None = None,
      session_id: str | None = None,
      now: float | None = None,
  ) -> CompactionCleanupStats:
    """Applies optional retention/eviction policies and returns delete counts.

    Implementations should support two policy knobs:
    - `max_age_seconds`: remove records older than `now - max_age_seconds`.
    - `max_records_per_kind`: keep at most this many newest records per kind.

    When `session_id` is set, cleanup should scope to artifacts that are tied to
    that session (`Observation`, `Reflection`, and `TaskStateAnchor`). Artifact
    kinds that are not session-addressable may be ignored for session-scoped
    cleanup.

    The default implementation is a no-op to preserve backward compatibility for
    services that do not yet implement retention controls.

    Args:
      max_age_seconds: Maximum age of retained records in seconds.
      max_records_per_kind: Per-kind record count limit to retain.
      session_id: Optional session scope for cleanup.
      now: Optional current timestamp override used for deterministic cleanup.

    Returns:
      A `CompactionCleanupStats` object containing per-kind delete counts.
    """
    del max_age_seconds, max_records_per_kind, session_id, now
    return CompactionCleanupStats()

  async def evict_session(self, session_id: str) -> CompactionCleanupStats:
    """Deletes all session-scoped artifacts for a single session.

    This method is intended for explicit targeted eviction of one session's
    `Observation`, `Reflection`, and `TaskStateAnchor` records.

    The default implementation is a no-op to preserve backward compatibility for
    services that do not yet implement targeted session eviction.

    Args:
      session_id: Session identifier whose session-scoped records should be
        removed.

    Returns:
      A `CompactionCleanupStats` object containing per-kind delete counts.
    """
    del session_id
    return CompactionCleanupStats()
