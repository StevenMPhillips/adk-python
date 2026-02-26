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

from google.adk.compaction.compactors.patch_compactor import PatchCompactor
from google.adk.compaction.compactors.registry import ToolRunCompactorRegistry
from google.adk.compaction.config import HybridEventsCompactionConfig
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)


def test_hybrid_events_compaction_config_defaults():
  config = HybridEventsCompactionConfig(
      compaction_service=InMemoryCompactionService(),
      compaction_interval=99,
      overlap_size=0,
  )

  assert isinstance(config.tool_run_compactor_registry, ToolRunCompactorRegistry)
  assert isinstance(config.patch_compactor, PatchCompactor)
  assert config.enable_deterministic_compaction is True
  assert config.enable_observational_memory is False
  assert config.enable_hybrid_prompt_assembly is False
  assert config.hybrid_prompt_token_budget == 8_000
  assert config.hybrid_raw_turns_count == 6
  assert config.hybrid_compactions_count == 6
  assert config.hybrid_observations_count == 3
  assert config.rehydration_evidence_token_budget == 2_000
