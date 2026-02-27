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

from google.adk.compaction.models import Decision
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import Observation
from google.adk.compaction.models import Reflection
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.storage.in_memory_compaction_service import InMemoryCompactionService
from google.adk.compaction.writers.reflection_writer import ReflectionWriter
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content
from google.genai.types import Part
import pytest


def _sample_evidence_ref(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='observation', ref_id=ref_id)


def _sample_item(text: str, ref_id: str) -> EvidencedItem:
  return EvidencedItem(text=text, evidence_refs=[_sample_evidence_ref(ref_id)])


def _sample_task_state() -> TaskStateAnchor:
  return TaskStateAnchor(
      session_id='session-1',
      state_version=3,
      objective='Improve observational memory quality.',
      constraints=[_sample_item('Only keep supported conclusions.', 'obs-1')],
      hypotheses=[_sample_item('Grouped windows improve reflection.', 'obs-1')],
      known_failures=[_sample_item('Unsupported facts cause drift.', 'obs-1')],
      current_plan=[
          _sample_item('Write one reflection per observation window.', 'obs-1')
      ],
      next_steps=[_sample_item('Add reflection writer tests.', 'obs-1')],
      last_updated_seq=50,
  )


def _sample_observation(index: int) -> Observation:
  observation_id = f'obs-{index}'
  return Observation(
      observation_id=observation_id,
      session_id='session-1',
      start_seq=index * 2,
      end_seq=index * 2 + 1,
      text=f'Observation {index} from recent work.',
      decisions=[
          Decision(
              text='Follow evidence-backed implementation steps.',
              kind='explicit',
              evidence_refs=[_sample_evidence_ref(observation_id)],
          )
      ],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[_sample_evidence_ref(observation_id)],
  )


def _sample_observations(count: int) -> list[Observation]:
  return [_sample_observation(i) for i in range(1, count + 1)]


def _sample_reflection_json() -> str:
  return Reflection(
      session_id='placeholder-session',
      covers_observation_ids=['obs-9', 'obs-10', 'obs-11', 'obs-12'],
      text='Hypothesis: strategy updates should focus on repeated failures.',
      stable_facts=[
          _sample_item(
              'Recent fixes depend on cited observation windows.', 'obs-10'
          )
      ],
      recurring_failures=[
          _sample_item(
              'Patches without tests repeatedly regress behavior.', 'obs-11'
          )
      ],
      strategy_updates=[
          _sample_item(
              'Require evidence refs for every reflection item.', 'obs-12'
          )
      ],
      evidence_refs=[_sample_evidence_ref('obs-12')],
  ).model_dump_json(by_alias=True)


def _sample_reflection_json_with_ref(ref_id: str) -> str:
  return Reflection(
      session_id='placeholder-session',
      covers_observation_ids=[ref_id],
      text='Hypothesis: focus on grounded observation ids.',
      stable_facts=[
          _sample_item(
              'Recent fixes depend on cited observation windows.',
              ref_id,
          )
      ],
      recurring_failures=[],
      strategy_updates=[],
      evidence_refs=[_sample_evidence_ref(ref_id)],
  ).model_dump_json(by_alias=True)


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestReflectionWriter(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_llm = AsyncMock(spec=BaseLlm)
    self.mock_llm.model = 'test-model'
    self.compaction_service = InMemoryCompactionService()
    self.writer = ReflectionWriter(
        llm=self.mock_llm,
        compaction_service=self.compaction_service,
    )

  def test_should_write_reflection_when_count_reaches_threshold(self):
    assert self.writer.should_write_reflection(
        recent_observations=_sample_observations(10)
    )
    assert self.writer.should_write_reflection(
        recent_observations=_sample_observations(11)
    )

  async def test_maybe_write_reflection_saves_valid_output(self):
    mock_llm_response = Mock(
        content=Content(parts=[Part(text=_sample_reflection_json())])
    )

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    reflection = await self.writer.maybe_write_reflection(
        recent_observations=_sample_observations(12),
        current_task_state=_sample_task_state(),
    )

    assert reflection is not None
    assert reflection.session_id == 'session-1'
    assert reflection.covers_observation_ids == [
        'obs-9',
        'obs-10',
        'obs-11',
        'obs-12',
    ]

    saved = await self.compaction_service.get_latest_reflection('session-1')
    assert saved == reflection

    self.mock_llm.generate_content_async.assert_called_once()
    args, kwargs = self.mock_llm.generate_content_async.call_args
    llm_request = args[0]
    assert isinstance(llm_request, LlmRequest)
    assert kwargs['stream'] is False
    contents = llm_request.contents or []
    assert contents
    parts = contents[0].parts or []
    assert parts
    prompt_text = parts[0].text or ''
    assert (
        'Every item in stableFacts, recurringFailures, and strategyUpdates'
        in prompt_text
    )
    assert 'label uncertain claims as hypotheses' in prompt_text

  async def test_maybe_write_reflection_drops_missing_item_evidence(self):
    invalid_reflection_json = Reflection(
        session_id='placeholder-session',
        covers_observation_ids=['obs-11'],
        text='invalid',
        stable_facts=[EvidencedItem(text='Unsupported item', evidence_refs=[])],
        recurring_failures=[],
        strategy_updates=[],
        evidence_refs=[],
    ).model_dump_json(by_alias=True)
    mock_llm_response = Mock(
        content=Content(parts=[Part(text=invalid_reflection_json)])
    )

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    reflection = await self.writer.maybe_write_reflection(
        recent_observations=_sample_observations(12),
        current_task_state=_sample_task_state(),
    )

    assert reflection is not None
    assert reflection.stable_facts == []

    saved = await self.compaction_service.get_latest_reflection('session-1')
    assert saved == reflection

  async def test_maybe_write_reflection_skips_unknown_evidence_refs(self):
    mock_llm_response = Mock(
        content=Content(
            parts=[Part(text=_sample_reflection_json_with_ref('obs-unknown'))]
        )
    )

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    reflection = await self.writer.maybe_write_reflection(
        recent_observations=_sample_observations(12),
        current_task_state=_sample_task_state(),
    )

    assert reflection is None

  async def test_maybe_write_reflection_canonicalizes_prefixed_refs(self):
    mock_llm_response = Mock(
        content=Content(
            parts=[Part(text=_sample_reflection_json_with_ref('observation:obs-12'))]
        )
    )

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    reflection = await self.writer.maybe_write_reflection(
        recent_observations=_sample_observations(12),
        current_task_state=_sample_task_state(),
    )

    assert reflection is not None
    assert reflection.covers_observation_ids == ['obs-12']
    assert reflection.stable_facts[0].evidence_refs[0].ref_id == 'obs-12'
