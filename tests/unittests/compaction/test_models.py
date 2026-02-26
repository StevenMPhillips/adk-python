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
from google.adk.compaction.models import FileLineRef
from google.adk.compaction.models import HunkSummary
from google.adk.compaction.models import PatchCompaction
from google.adk.compaction.models import Provenance
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
