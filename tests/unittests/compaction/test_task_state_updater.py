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

import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.compaction.models import CompactionStats
from google.adk.compaction.models import Decision
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import Observation
from google.adk.compaction.models import Provenance
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.models import ToolRunCompaction
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)
from google.adk.compaction.writers.task_state_updater import TaskStateUpdater
from google.adk.models.base_llm import BaseLlm
from google.genai.types import Content
from google.genai.types import Part
import pytest


def _sample_ref(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='event', ref_id=ref_id)


def _sample_item(text: str, ref_id: str = 'evt-1') -> EvidencedItem:
  return EvidencedItem(text=text, evidence_refs=[_sample_ref(ref_id)])


def _sample_task_state() -> TaskStateAnchor:
  return TaskStateAnchor(
      session_id='session-1',
      state_version=4,
      objective='Fix failing compaction pipeline tests.',
      constraints=[
          _sample_item('Keep all updates evidence-backed.', 'evt-base')
      ],
      hypotheses=[
          _sample_item('Smaller context windows reduce drift.', 'evt-base')
      ],
      known_failures=[
          _sample_item('Parser misses nested aliases.', 'evt-base')
      ],
      current_plan=[_sample_item('Implement task-state updater.', 'evt-base')],
      next_steps=[_sample_item('Add updater unit tests.', 'evt-base')],
      last_updated_seq=30,
  )


def _sample_observation() -> Observation:
  return Observation(
      observation_id='obs-11',
      session_id='session-1',
      start_seq=31,
      end_seq=36,
      text='Observed explicit and inferred decisions from recent turns.',
      decisions=[
          Decision(
              text='Pin task-state output format to schema.',
              kind='explicit',
              evidence_refs=[_sample_ref('evt-1')],
          ),
          Decision(
              text='Retries might hide root causes.',
              kind='inferred',
              evidence_refs=[_sample_ref('evt-2')],
          ),
      ],
      learned_constraints=[
          Decision(
              text='Keep all state constraints grounded in evidence.',
              kind='explicit',
              evidence_refs=[_sample_ref('evt-1')],
          )
      ],
      open_questions=[],
      next_steps=[
          _sample_item('Integrate updater into runner hooks.', 'evt-2')
      ],
      evidence_refs=[_sample_ref('evt-1'), _sample_ref('evt-2')],
  )


def _sample_tool_run_compaction() -> ToolRunCompaction:
  return ToolRunCompaction(
      event_id='tool-evt-1',
      compaction_version=1,
      command='pytest tests/unittests/compaction/test_task_state_updater.py',
      exit_code=1,
      duration_ms=200,
      error_signatures=['pytest::AssertionError::test_task_state'],
      key_errors=['AssertionError: expected evidence-backed constraint'],
      tests_failed=['tests/unittests/compaction/test_task_state_updater.py::x'],
      file_line_refs=[],
      trimmed_trace=['E AssertionError: expected evidence-backed constraint'],
      salient_snippets=['assert ref_id in allowed_ids'],
      stats=CompactionStats(
          raw_tokens_est=300,
          compact_tokens_est=40,
          compression_ratio=7.5,
      ),
      provenance=Provenance(event_id='tool-evt-1'),
  )


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestTaskStateUpdater(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.compaction_service = InMemoryCompactionService()
    self.updater = TaskStateUpdater(
        compaction_service=self.compaction_service,
        task_state_token_budget=500,
    )

  async def test_update_from_observation_moves_inferred_items_to_hypotheses(
      self,
  ):
    observation = _sample_observation()

    updated_state = await self.updater.update_from_observation(
        observation=observation,
        current_task_state=_sample_task_state(),
    )

    assert updated_state.state_version == 5
    assert updated_state.last_updated_seq == 36
    assert 'Retries might hide root causes.' in [
        item.text for item in updated_state.hypotheses
    ]
    assert 'Retries might hide root causes.' not in [
        item.text for item in updated_state.constraints
    ]

    saved_state = await self.compaction_service.get_task_state('session-1')
    assert saved_state == updated_state

  async def test_update_from_observation_requires_constraint_provenance(self):
    observation = _sample_observation().model_copy(deep=True)
    observation.learned_constraints = [
        Decision(
            text='Never run full test suites during edits.',
            kind='explicit',
            evidence_refs=[],
        )
    ]

    with pytest.raises(
        ValueError,
        match='Constraints must include evidence_refs or explicit user',
    ):
      await self.updater.update_from_observation(
          observation=observation,
          current_task_state=_sample_task_state(),
      )

  async def test_update_from_observation_allows_explicit_user_instruction(self):
    explicit_instruction = 'Never run full test suites during edits.'
    observation = _sample_observation().model_copy(deep=True)
    observation.learned_constraints = [
        Decision(
            text=explicit_instruction,
            kind='explicit',
            evidence_refs=[],
        )
    ]

    updated_state = await self.updater.update_from_observation(
        observation=observation,
        current_task_state=_sample_task_state(),
        explicit_user_instructions=[explicit_instruction],
    )

    assert explicit_instruction in [
        item.text for item in updated_state.constraints
    ]

  async def test_update_from_observation_with_llm_validates_evidence_refs(self):
    mock_llm = AsyncMock(spec=BaseLlm)
    mock_llm.model = 'test-model'
    updater = TaskStateUpdater(
        compaction_service=self.compaction_service,
        llm=mock_llm,
    )
    llm_task_state = _sample_task_state().model_copy(deep=True)
    llm_task_state.constraints = [
        EvidencedItem(
            text='Constraint produced by LLM with bad provenance.',
            evidence_refs=[EvidenceRef(ref_type='event', ref_id='evt-unknown')],
        )
    ]
    llm_task_state_json = llm_task_state.model_dump_json(by_alias=True)
    mock_llm_response = Mock(
        content=Content(parts=[Part(text=llm_task_state_json)])
    )

    async def async_gen():
      yield mock_llm_response

    mock_llm.generate_content_async.return_value = async_gen()

    with pytest.raises(
        ValueError,
        match='Constraint evidence ref id is not in allowed evidence',
    ):
      await updater.update_from_observation(
          observation=_sample_observation(),
          current_task_state=_sample_task_state(),
          use_llm=True,
      )

  async def test_update_from_tool_run_bumps_version_and_persists(self):
    updated_state = await self.updater.update_from_tool_run(
        tool_run_compaction=_sample_tool_run_compaction(),
        current_task_state=_sample_task_state(),
    )

    assert updated_state.state_version == 5
    assert 'Error signature: pytest::AssertionError::test_task_state' in [
        item.text for item in updated_state.known_failures
    ]
    assert 'AssertionError: expected evidence-backed constraint' in [
        item.text for item in updated_state.hypotheses
    ]

    saved_state = await self.compaction_service.get_task_state('session-1')
    assert saved_state == updated_state

  async def test_update_applies_token_budget_cap(self):
    capped_updater = TaskStateUpdater(
        compaction_service=self.compaction_service,
        task_state_token_budget=70,
    )
    observation = _sample_observation().model_copy(deep=True)
    observation.next_steps = [
        _sample_item(
            text=(
                'Very long step requiring additional context and details '
                f'index {index}'
            ),
            ref_id='evt-2',
        )
        for index in range(20)
    ]

    updated_state = await capped_updater.update_from_observation(
        observation=observation,
        current_task_state=_sample_task_state(),
    )

    token_estimate = len(updated_state.model_dump_json(by_alias=True)) // 4
    assert token_estimate <= 70
