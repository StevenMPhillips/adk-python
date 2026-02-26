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

import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from pydantic import ValidationError

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
from google.adk.compaction.writers.reflection_writer import ReflectionWriter
from google.adk.models.base_llm import BaseLlm
from google.genai.types import Content
from google.genai.types import Part
import pytest


def _sample_observation_ref(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='observation', ref_id=ref_id)


def _sample_event_ref(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='event', ref_id=ref_id)


def _sample_item(text: str, ref: EvidenceRef) -> EvidencedItem:
  return EvidencedItem(text=text, evidence_refs=[ref])


def _sample_turns() -> list[RawTurn]:
  return [
      RawTurn(
          session_id='session-1',
          seq=10,
          event_id='evt-1',
          author='user',
          text='Investigate observation quality.',
          raw_tokens_est=40,
      ),
      RawTurn(
          session_id='session-1',
          seq=11,
          event_id='evt-2',
          author='model',
          text='I will produce evidence-backed output.',
          raw_tokens_est=40,
      ),
  ]


def _sample_task_state() -> TaskStateAnchor:
  return TaskStateAnchor(
      session_id='session-1',
      state_version=1,
      objective='Harden evidence validation behavior.',
      constraints=[_sample_item('Use cited evidence only.', _sample_event_ref('evt-1'))],
      hypotheses=[_sample_item('Malformed refs should fail fast.', _sample_event_ref('evt-1'))],
      known_failures=[_sample_item('Silent acceptance causes drift.', _sample_event_ref('evt-1'))],
      current_plan=[_sample_item('Add failure-mode tests.', _sample_event_ref('evt-1'))],
      next_steps=[_sample_item('Validate both writers.', _sample_event_ref('evt-1'))],
      last_updated_seq=20,
  )


def _sample_observations() -> list[Observation]:
  observations: list[Observation] = []
  for index in range(1, 13):
    observation_id = f'obs-{index}'
    observations.append(
        Observation(
            observation_id=observation_id,
            session_id='session-1',
            start_seq=index * 2,
            end_seq=index * 2 + 1,
            text=f'Observation {index}',
            decisions=[
                Decision(
                    text='Keep reflection evidence-backed.',
                    kind='explicit',
                    evidence_refs=[_sample_observation_ref(observation_id)],
                )
            ],
            learned_constraints=[],
            open_questions=[],
            next_steps=[],
            evidence_refs=[_sample_observation_ref(observation_id)],
        )
    )
  return observations


def _llm_response_from_json(raw_json: str) -> Mock:
  return Mock(content=Content(parts=[Part(text=raw_json)]))


def _configure_single_llm_response(mock_llm: AsyncMock, raw_json: str) -> None:
  async def async_gen():
    yield _llm_response_from_json(raw_json)

  mock_llm.generate_content_async.return_value = async_gen()


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestWriterEvidenceValidation(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_llm = AsyncMock(spec=BaseLlm)
    self.mock_llm.model = 'test-model'
    self.compaction_service = InMemoryCompactionService()
    self.observation_writer = ObservationWriter(
        llm=self.mock_llm,
        compaction_service=self.compaction_service,
        raw_token_threshold=1,
    )
    self.reflection_writer = ReflectionWriter(
        llm=self.mock_llm,
        compaction_service=self.compaction_service,
    )

  async def test_observation_writer_rejects_malformed_evidence_ref_structure(self):
    invalid_json = json.dumps({
        'observationId': 'obs-generated',
        'sessionId': 'placeholder',
        'startSeq': 0,
        'endSeq': 0,
        'text': 'Malformed decision evidence ref.',
        'decisions': [
            {
                'text': 'Missing refId should fail model validation.',
                'kind': 'explicit',
                'evidenceRefs': [{'refType': 'event'}],
            }
        ],
        'learnedConstraints': [],
        'openQuestions': [],
        'nextSteps': [],
        'evidenceRefs': [{'refType': 'event', 'refId': 'evt-1'}],
    })
    _configure_single_llm_response(self.mock_llm, invalid_json)

    with pytest.raises(ValidationError, match='refId'):
      await self.observation_writer.maybe_write_observation(
          recent_raw_turns=_sample_turns(),
          recent_tool_run_compactions=[],
          recent_patch_compactions=[],
          current_task_state=_sample_task_state(),
          episode_closed=True,
      )

  async def test_observation_writer_rejects_unknown_top_level_evidence_ref(self):
    invalid_json = json.dumps({
        'observationId': 'obs-generated',
        'sessionId': 'placeholder',
        'startSeq': 0,
        'endSeq': 0,
        'text': 'Top-level evidence refs must be allowed.',
        'decisions': [
            {
                'text': 'This decision cites a valid ref.',
                'kind': 'inferred',
                'evidenceRefs': [{'refType': 'event', 'refId': 'evt-1'}],
            }
        ],
        'learnedConstraints': [],
        'openQuestions': [],
        'nextSteps': [],
        'evidenceRefs': [{'refType': 'event', 'refId': 'evt-unknown'}],
    })
    _configure_single_llm_response(self.mock_llm, invalid_json)

    with pytest.raises(
        ValueError,
        match='Top-level evidence ref id is not in allowed evidence set.',
    ):
      await self.observation_writer.maybe_write_observation(
          recent_raw_turns=_sample_turns(),
          recent_tool_run_compactions=[],
          recent_patch_compactions=[],
          current_task_state=_sample_task_state(),
          episode_closed=True,
      )

  async def test_reflection_writer_rejects_non_list_item_evidence_refs(self):
    invalid_json = json.dumps({
        'reflectionId': 'reflection-generated',
        'sessionId': 'placeholder',
        'coversObservationIds': ['obs-9', 'obs-10', 'obs-11', 'obs-12'],
        'text': 'Malformed evidenceRefs field type.',
        'stableFacts': [
            {
                'text': 'This field uses invalid evidenceRefs shape.',
                'evidenceRefs': 'obs-10',
            }
        ],
        'recurringFailures': [],
        'strategyUpdates': [],
        'evidenceRefs': [{'refType': 'observation', 'refId': 'obs-10'}],
    })
    _configure_single_llm_response(self.mock_llm, invalid_json)

    with pytest.raises(ValidationError, match='evidenceRefs'):
      await self.reflection_writer.maybe_write_reflection(
          recent_observations=_sample_observations(),
          current_task_state=_sample_task_state(),
      )

  async def test_reflection_writer_rejects_unknown_top_level_evidence_ref(self):
    invalid_json = json.dumps({
        'reflectionId': 'reflection-generated',
        'sessionId': 'placeholder',
        'coversObservationIds': ['obs-9', 'obs-10', 'obs-11', 'obs-12'],
        'text': 'Unknown top-level observation ref should fail.',
        'stableFacts': [
            {
                'text': 'Use valid per-item evidence.',
                'evidenceRefs': [{'refType': 'observation', 'refId': 'obs-10'}],
            }
        ],
        'recurringFailures': [],
        'strategyUpdates': [],
        'evidenceRefs': [{'refType': 'observation', 'refId': 'obs-unknown'}],
    })
    _configure_single_llm_response(self.mock_llm, invalid_json)

    with pytest.raises(
        ValueError,
        match='Top-level evidence ref id is not in allowed observation ids.',
    ):
      await self.reflection_writer.maybe_write_reflection(
          recent_observations=_sample_observations(),
          current_task_state=_sample_task_state(),
      )
