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
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import Observation
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)
from google.adk.compaction.writers.observation_writer import ObservationWriter
from google.adk.compaction.writers.observation_writer import RawTurn
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content
from google.genai.types import Part
import pytest


def _sample_evidence_ref(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='event', ref_id=ref_id)


def _sample_item(text: str, ref_id: str) -> EvidencedItem:
  return EvidencedItem(text=text, evidence_refs=[_sample_evidence_ref(ref_id)])


def _sample_task_state() -> TaskStateAnchor:
  return TaskStateAnchor(
      session_id='session-1',
      state_version=1,
      objective='Stabilize the observation writer.',
      constraints=[_sample_item('Keep outputs evidence-backed.', 'evt-1')],
      hypotheses=[_sample_item('Structured prompts reduce drift.', 'evt-1')],
      known_failures=[_sample_item('Missing refs break trust.', 'evt-1')],
      current_plan=[_sample_item('Generate one observation per window.', 'evt-1')],
      next_steps=[_sample_item('Add writer tests.', 'evt-1')],
      last_updated_seq=22,
  )


def _sample_turns() -> list[RawTurn]:
  return [
      RawTurn(
          session_id='session-1',
          seq=10,
          event_id='evt-1',
          author='user',
          text='Please fix observation storage.',
          raw_tokens_est=80,
      ),
      RawTurn(
          session_id='session-1',
          seq=12,
          event_id='evt-2',
          author='model',
          text='I will add a writer and tests.',
          raw_tokens_est=90,
      ),
  ]


def _sample_observation_json(ref_id: str = 'evt-1') -> str:
  return Observation(
      session_id='placeholder-session',
      start_seq=0,
      end_seq=0,
      text='Observed implementation progress with test coverage.',
      decisions=[
          Decision(
              text='Implement an Observation writer and unit tests.',
              kind='explicit',
              evidence_refs=[_sample_evidence_ref(ref_id)],
          )
      ],
      learned_constraints=[
          Decision(
              text='Keep output schema strict and evidence-backed.',
              kind='inferred',
              evidence_refs=[_sample_evidence_ref(ref_id)],
          )
      ],
      open_questions=[_sample_item('Should we add retry policy?', ref_id)],
      next_steps=[_sample_item('Integrate with processor flow.', ref_id)],
      evidence_refs=[_sample_evidence_ref(ref_id)],
  ).model_dump_json(by_alias=True)


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestObservationWriter(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_llm = AsyncMock(spec=BaseLlm)
    self.mock_llm.model = 'test-model'
    self.compaction_service = InMemoryCompactionService()
    self.writer = ObservationWriter(
        llm=self.mock_llm,
        compaction_service=self.compaction_service,
        raw_token_threshold=150,
    )

  def test_should_write_observation_by_threshold_or_episode_close(self):
    turns = _sample_turns()

    assert self.writer.should_write_observation(
        recent_raw_turns=turns,
        episode_closed=False,
    )
    assert self.writer.should_write_observation(
        recent_raw_turns=[
            RawTurn(
                session_id='session-1',
                seq=1,
                event_id='evt-small',
                author='user',
                text='short',
                raw_tokens_est=2,
            )
        ],
        episode_closed=True,
    )
    assert not self.writer.should_write_observation(
        recent_raw_turns=[],
        episode_closed=True,
    )

  async def test_maybe_write_observation_saves_valid_output(self):
    mock_llm_response = Mock(
        content=Content(parts=[Part(text=_sample_observation_json())])
    )

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    observation = await self.writer.maybe_write_observation(
        recent_raw_turns=_sample_turns(),
        recent_tool_run_compactions=[],
        recent_patch_compactions=[],
        current_task_state=_sample_task_state(),
        episode_closed=False,
    )

    assert observation is not None
    assert observation.session_id == 'session-1'
    assert observation.start_seq == 10
    assert observation.end_seq == 12

    saved = await self.compaction_service.get_observations('session-1')
    assert saved == [observation]

    self.mock_llm.generate_content_async.assert_called_once()
    args, kwargs = self.mock_llm.generate_content_async.call_args
    llm_request = args[0]
    assert isinstance(llm_request, LlmRequest)
    assert kwargs['stream'] is False
    prompt_text = llm_request.contents[0].parts[0].text
    assert 'Every item in decisions and learnedConstraints' in prompt_text
    assert 'Set decision.kind to "explicit" only' in prompt_text

  async def test_maybe_write_observation_rejects_missing_evidence_refs(self):
    invalid_observation = Observation(
        session_id='placeholder-session',
        start_seq=0,
        end_seq=0,
        text='invalid',
        decisions=[
            Decision(
                text='Missing evidence.',
                kind='inferred',
                evidence_refs=[],
            )
        ],
        learned_constraints=[],
        open_questions=[],
        next_steps=[],
        evidence_refs=[],
    ).model_dump_json(by_alias=True)
    mock_llm_response = Mock(content=Content(parts=[Part(text=invalid_observation)]))

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    with pytest.raises(
        ValueError, match='Each decision/constraint must include evidence_refs.'
    ):
      await self.writer.maybe_write_observation(
          recent_raw_turns=_sample_turns(),
          recent_tool_run_compactions=[],
          recent_patch_compactions=[],
          current_task_state=_sample_task_state(),
          episode_closed=True,
      )

    saved = await self.compaction_service.get_observations('session-1')
    assert saved == []

  async def test_maybe_write_observation_rejects_unknown_evidence_refs(self):
    mock_llm_response = Mock(
        content=Content(parts=[Part(text=_sample_observation_json('evt-unknown'))])
    )

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    with pytest.raises(
        ValueError, match='Decision evidence ref id is not in allowed evidence'
    ):
      await self.writer.maybe_write_observation(
          recent_raw_turns=_sample_turns(),
          recent_tool_run_compactions=[],
          recent_patch_compactions=[],
          current_task_state=_sample_task_state(),
      )
