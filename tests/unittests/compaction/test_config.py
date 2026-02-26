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
from google.adk.compaction.storage.in_memory_compaction_service import InMemoryCompactionService
from pydantic import ValidationError
import pytest


def test_hybrid_events_compaction_config_defaults():
  config = HybridEventsCompactionConfig(
      compaction_service=InMemoryCompactionService(),
      compaction_interval=99,
      overlap_size=0,
  )

  assert isinstance(
      config.tool_run_compactor_registry, ToolRunCompactorRegistry
  )
  assert isinstance(config.patch_compactor, PatchCompactor)
  assert config.enable_deterministic_compaction is True
  assert config.enable_observational_memory is False
  assert config.enable_hybrid_prompt_assembly is False
  assert config.hybrid_prompt_token_budget == 8_000
  assert config.hybrid_raw_turns_count == 6
  assert config.hybrid_compactions_count == 6
  assert config.hybrid_observations_count == 3
  assert config.rehydration_evidence_token_budget == 2_000


@pytest.mark.parametrize(
    'kwargs',
    [
        {'hybrid_prompt_token_budget': 0},
        {'hybrid_raw_turns_count': -1},
        {'hybrid_compactions_count': -1},
        {'hybrid_observations_count': -1},
        {'rehydration_evidence_token_budget': -1},
    ],
)
def test_hybrid_events_compaction_config_rejects_invalid_minimums(kwargs):
  with pytest.raises(ValidationError):
    HybridEventsCompactionConfig(
        compaction_service=InMemoryCompactionService(),
        compaction_interval=99,
        overlap_size=0,
        **kwargs,
    )


def test_hybrid_events_compaction_config_rejects_hybrid_without_deterministic():
  with pytest.raises(
      ValidationError,
      match='enable_hybrid_prompt_assembly requires '
      'enable_deterministic_compaction',
  ):
    HybridEventsCompactionConfig(
        compaction_service=InMemoryCompactionService(),
        compaction_interval=99,
        overlap_size=0,
        enable_deterministic_compaction=False,
        enable_hybrid_prompt_assembly=True,
        hybrid_compactions_count=0,
    )


def test_hybrid_events_compaction_config_rejects_compactions_without_deterministic(
):
  with pytest.raises(
      ValidationError,
      match='hybrid_compactions_count must be 0 when '
      'enable_deterministic_compaction is False',
  ):
    HybridEventsCompactionConfig(
        compaction_service=InMemoryCompactionService(),
        compaction_interval=99,
        overlap_size=0,
        enable_deterministic_compaction=False,
        hybrid_compactions_count=1,
    )


def test_hybrid_events_compaction_config_rejects_empty_hybrid_sources():
  with pytest.raises(
      ValidationError,
      match='enable_hybrid_prompt_assembly requires at least one non-zero '
      'hybrid count',
  ):
    HybridEventsCompactionConfig(
        compaction_service=InMemoryCompactionService(),
        compaction_interval=99,
        overlap_size=0,
        enable_hybrid_prompt_assembly=True,
        hybrid_raw_turns_count=0,
        hybrid_compactions_count=0,
        hybrid_observations_count=0,
    )


def test_hybrid_events_compaction_config_rejects_rehydration_budget_over_prompt_budget(
):
  with pytest.raises(
      ValidationError,
      match='rehydration_evidence_token_budget must be <= '
      'hybrid_prompt_token_budget',
  ):
    HybridEventsCompactionConfig(
        compaction_service=InMemoryCompactionService(),
        compaction_interval=99,
        overlap_size=0,
        hybrid_prompt_token_budget=1_000,
        rehydration_evidence_token_budget=1_001,
    )


def test_hybrid_events_compaction_config_accepts_consistent_hybrid_settings():
  config = HybridEventsCompactionConfig(
      compaction_service=InMemoryCompactionService(),
      compaction_interval=99,
      overlap_size=0,
      enable_hybrid_prompt_assembly=True,
      hybrid_prompt_token_budget=5_000,
      hybrid_raw_turns_count=2,
      hybrid_compactions_count=2,
      hybrid_observations_count=1,
      rehydration_evidence_token_budget=1_000,
  )

  assert config.enable_hybrid_prompt_assembly is True
