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

from google.adk.compaction.models import CompactionStats
from google.adk.compaction.models import Decision
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import FileLineRef
from google.adk.compaction.models import HunkSummary
from google.adk.compaction.models import Observation
from google.adk.compaction.models import PatchCompaction
from google.adk.compaction.models import Provenance
from google.adk.compaction.models import Reflection
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.models import ToolRunCompaction
from pydantic import ValidationError
import pytest


def _sample_stats() -> CompactionStats:
  return CompactionStats(
      raw_tokens_est=400,
      compact_tokens_est=100,
      compression_ratio=4.0,
  )


def _sample_provenance() -> Provenance:
  return Provenance(
      event_id='evt-1',
      stdout_offsets=(5, 18),
      stderr_offsets=(20, 25),
  )


def _sample_evidence_ref() -> EvidenceRef:
  return EvidenceRef(ref_type='event', ref_id='evt-1')


def _sample_evidenced_item(text: str) -> EvidencedItem:
  return EvidencedItem(text=text, evidence_refs=[_sample_evidence_ref()])


def test_tool_run_compaction_json_round_trip_uses_camel_case_aliases():
  model = ToolRunCompaction(
      event_id='evt-1',
      compaction_version=1,
      command='pytest tests/unittests/compaction/test_models.py',
      exit_code=1,
      duration_ms=2_500,
      error_signatures=['AssertionError'],
      key_errors=['failed to parse config'],
      tests_failed=['tests/unittests/compaction/test_models.py::test_case'],
      file_line_refs=[FileLineRef(path='src/foo.py', line=12, col=4)],
      trimmed_trace=['Traceback (most recent call last):'],
      salient_snippets=['assert value == expected'],
      stats=_sample_stats(),
      provenance=_sample_provenance(),
  )

  payload = model.model_dump_json(by_alias=True)
  restored = ToolRunCompaction.model_validate_json(payload)
  payload_dict = json.loads(payload)

  assert restored == model
  assert 'eventId' in payload_dict
  assert 'durationMs' in payload_dict
  assert 'fileLineRefs' in payload_dict
  assert 'rawTokensEst' in payload_dict['stats']
  assert 'stdoutOffsets' in payload_dict['provenance']


def test_patch_compaction_json_round_trip_supports_alias_validation():
  payload = {
      'eventId': 'evt-2',
      'compactionVersion': 1,
      'filesChanged': ['src/google/adk/compaction/models.py'],
      'hunks': [
          {
              'path': 'src/google/adk/compaction/models.py',
              'anchorBefore': 'class Existing:',
              'anchorAfter': 'class Added:',
              'snippet': '+class Added:\n+  pass',
          }
      ],
      'semanticTags': ['refactor', 'schema'],
      'stats': {
          'rawTokensEst': 80,
          'compactTokensEst': 30,
          'compressionRatio': 2.6,
      },
      'provenance': {'eventId': 'evt-2'},
  }

  model = PatchCompaction.model_validate(payload)
  restored = PatchCompaction.model_validate_json(
      model.model_dump_json(by_alias=True)
  )

  assert model == restored
  assert model.hunks == [
      HunkSummary(
          path='src/google/adk/compaction/models.py',
          anchor_before='class Existing:',
          anchor_after='class Added:',
          snippet='+class Added:\n+  pass',
      )
  ]


def test_observation_json_round_trip_uses_camel_case_aliases():
  model = Observation(
      session_id='session-1',
      start_seq=10,
      end_seq=22,
      text='Observed repeated failures while running tests.',
      decisions=[
          Decision(
              text='Retry with a narrower test target.',
              kind='explicit',
              evidence_refs=[_sample_evidence_ref()],
          )
      ],
      learned_constraints=[
          Decision(
              text='Avoid full-suite runs during active debugging.',
              kind='inferred',
              evidence_refs=[_sample_evidence_ref()],
          )
      ],
      open_questions=[
          _sample_evidenced_item('Why does the parser fail on aliases?')
      ],
      next_steps=[_sample_evidenced_item('Add focused regression tests.')],
      evidence_refs=[_sample_evidence_ref()],
  )

  payload = model.model_dump_json(by_alias=True)
  restored = Observation.model_validate_json(payload)
  payload_dict = json.loads(payload)

  assert restored == model
  assert 'observationId' in payload_dict
  assert 'sessionId' in payload_dict
  assert 'startSeq' in payload_dict
  assert 'learnedConstraints' in payload_dict
  assert payload_dict['decisions'][0]['kind'] == 'explicit'
  assert (
      payload_dict['openQuestions'][0]['evidenceRefs'][0]['refType']
      == 'event'
  )


def test_reflection_json_round_trip_supports_alias_validation():
  payload = {
      'reflectionId': 'refl-1',
      'sessionId': 'session-1',
      'coversObservationIds': ['obs-1', 'obs-2'],
      'text': 'Refined strategy after repeated parser errors.',
      'stableFacts': [
          {
              'text': 'Alias parsing is stable for known keys.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-2'}],
          }
      ],
      'recurringFailures': [
          {
              'text': 'Unknown fields fail with ValidationError.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-3'}],
          }
      ],
      'strategyUpdates': [
          {
              'text': 'Preserve strict extra field validation.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-4'}],
          }
      ],
      'evidenceRefs': [{'refType': 'event', 'refId': 'evt-1'}],
  }

  model = Reflection.model_validate(payload)
  restored = Reflection.model_validate_json(
      model.model_dump_json(by_alias=True)
  )

  assert model == restored
  assert model.strategy_updates == [
      EvidencedItem(
          text='Preserve strict extra field validation.',
          evidence_refs=[EvidenceRef(ref_type='event', ref_id='evt-4')],
      )
  ]


def test_task_state_anchor_json_round_trip_supports_alias_validation():
  payload = {
      'sessionId': 'session-9',
      'stateVersion': 1,
      'objective': 'Stabilize observational compaction output.',
      'constraints': [
          {
              'text': 'Keep schema strict with extra=forbid.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-5'}],
          }
      ],
      'hypotheses': [
          {
              'text': 'Evidence-linked items reduce hallucinations.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-6'}],
          }
      ],
      'knownFailures': [
          {
              'text': 'Missing aliases can break JSON consumers.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-7'}],
          }
      ],
      'currentPlan': [
          {
              'text': 'Add model classes and round-trip tests.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-8'}],
          }
      ],
      'nextSteps': [
          {
              'text': 'Run focused compaction model tests.',
              'evidenceRefs': [{'refType': 'event', 'refId': 'evt-9'}],
          }
      ],
      'lastUpdatedSeq': 42,
  }

  model = TaskStateAnchor.model_validate(payload)
  restored = TaskStateAnchor.model_validate_json(
      model.model_dump_json(by_alias=True)
  )
  payload_dict = json.loads(model.model_dump_json(by_alias=True))

  assert model == restored
  assert payload_dict['stateVersion'] == 1
  assert payload_dict['currentPlan'][0]['evidenceRefs'][0]['refId'] == 'evt-8'


@pytest.mark.parametrize(
    'model_type, payload',
    [
        (FileLineRef, {'path': 'src/foo.py', 'line': 3, 'unknown': 'x'}),
        (
            Provenance,
            {'eventId': 'evt-9', 'stderrOffsets': [1, 2], 'extra': 'x'},
        ),
    ],
)
def test_compaction_models_forbid_unknown_fields(model_type, payload):
  with pytest.raises(ValidationError):
    model_type.model_validate(payload)
