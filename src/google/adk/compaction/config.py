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

from pydantic import Field

from ..apps.app import EventsCompactionConfig
from .compactors.patch_compactor import PatchCompactor
from .compactors.registry import ToolRunCompactorRegistry
from .storage.base_compaction_service import BaseCompactionService


class HybridEventsCompactionConfig(EventsCompactionConfig):
  """Events compaction config with deterministic artifact compaction."""

  compaction_service: BaseCompactionService
  """Storage backend for deterministic compaction artifacts."""

  tool_run_compactor_registry: ToolRunCompactorRegistry = Field(
      default_factory=ToolRunCompactorRegistry
  )
  """Registry for selecting tool-run compactors."""

  patch_compactor: PatchCompactor = Field(default_factory=PatchCompactor)
  """Compactor for patch and diff outputs."""

  enable_deterministic_compaction: bool = True
  """Whether deterministic compaction hooks run inline on tool events."""

  enable_observational_memory: bool = False
  """Whether observational-memory compaction is enabled."""
