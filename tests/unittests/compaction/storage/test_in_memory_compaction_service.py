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

from google.adk.compaction.models import CompactionStats
from google.adk.compaction.models import Decision
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import FileLineRef
from google.adk.compaction.models import HunkSummary
from google.adk.compaction.models import Observation
from google.adk.compaction.models import PatchCompaction
from google.adk.compaction.models import Provenance
from google.adk.compaction.models import Reflection
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.models import ToolRunCompaction
from google.adk.compaction.storage.in_memory_compaction_service import InMemoryCompactionService
import pytest


def _sample_stats() -> CompactionStats:
  return CompactionStats(
      raw_tokens_est=100,
      compact_tokens_est=30,
      compression_ratio=3.3,
  )


def _sample_provenance(event_id: str) -> Provenance:
  return Provenance(event_id=event_id)


def _sample_evidence(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='event', ref_id=ref_id)


def _sample_item(text: str, ref_id: str) -> EvidencedItem:
  return EvidencedItem(text=text, evidence_refs=[_sample_evidence(ref_id)])


@pytest.mark.asyncio
async def test_save_and_get_all_artifact_types():
  service = InMemoryCompactionService()

  tool_run = ToolRunCompaction(
      event_id='evt-tool-1',
      compaction_version=1,
      command='pytest tests/unittests/compaction/storage',
      exit_code=1,
      duration_ms=250,
      error_signatures=['AssertionError', 'ValidationError'],
      key_errors=['assert x == y'],
      tests_failed=['test_file.py::test_case'],
      file_line_refs=[FileLineRef(path='src/foo.py', line=10, col=3)],
      trimmed_trace=['Traceback...'],
      salient_snippets=['x != y'],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-1'),
  )
  patch = PatchCompaction(
      event_id='evt-patch-1',
      compaction_version=1,
      files_changed=['src/foo.py'],
      hunks=[
          HunkSummary(
              path='src/foo.py',
              anchor_before='def old():',
              anchor_after='def new():',
              snippet='-old\n+new',
          )
      ],
      semantic_tags=['bugfix'],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-patch-1'),
  )
  observation = Observation(
      observation_id='obs-1',
      session_id='session-1',
      start_seq=1,
      end_seq=5,
      text='Observed flaky tests.',
      decisions=[
          Decision(
              text='Run targeted tests first.',
              kind='explicit',
              evidence_refs=[_sample_evidence('evt-tool-1')],
          )
      ],
      learned_constraints=[
          Decision(
              text='Avoid full suite while iterating.',
              kind='inferred',
              evidence_refs=[_sample_evidence('evt-tool-1')],
          )
      ],
      open_questions=[_sample_item('Why is this flaky?', 'evt-tool-1')],
      next_steps=[_sample_item('Add regression test.', 'evt-patch-1')],
      evidence_refs=[_sample_evidence('evt-tool-1')],
  )
  reflection = Reflection(
      reflection_id='refl-1',
      session_id='session-1',
      covers_observation_ids=['obs-1'],
      text='Prefer tight feedback loops.',
      stable_facts=[_sample_item('Targeted tests are faster.', 'evt-tool-1')],
      recurring_failures=[_sample_item('Flaky parser tests.', 'evt-tool-1')],
      strategy_updates=[_sample_item('Keep changes small.', 'evt-patch-1')],
      evidence_refs=[_sample_evidence('evt-tool-1')],
  )
  task_state = TaskStateAnchor(
      session_id='session-1',
      state_version=2,
      objective='Stabilize compaction storage.',
      constraints=[_sample_item('Keep methods async.', 'evt-tool-1')],
      hypotheses=[_sample_item('Simple dicts are enough.', 'evt-patch-1')],
      known_failures=[_sample_item('Schema drift risk.', 'evt-tool-1')],
      current_plan=[
          _sample_item('Implement in-memory service.', 'evt-patch-1')
      ],
      next_steps=[_sample_item('Add focused tests.', 'evt-patch-1')],
      last_updated_seq=8,
  )

  await service.save_tool_run_compaction(tool_run)
  await service.save_patch_compaction(patch)
  await service.save_observation(observation)
  await service.save_reflection(reflection)
  await service.save_task_state(task_state)

  assert await service.get_tool_run_compaction('evt-tool-1') == tool_run
  assert await service.get_patch_compaction('evt-patch-1') == patch
  assert await service.get_observations('session-1') == [observation]
  assert await service.get_latest_reflection('session-1') == reflection
  assert await service.get_task_state('session-1') == task_state


@pytest.mark.asyncio
async def test_query_by_error_signature_returns_matching_tool_runs():
  service = InMemoryCompactionService()

  matching = ToolRunCompaction(
      event_id='evt-1',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-1'),
  )
  non_matching = ToolRunCompaction(
      event_id='evt-2',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['TypeError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-2'),
  )

  await service.save_tool_run_compaction(matching)
  await service.save_tool_run_compaction(non_matching)

  assert await service.query_by_error_signature('AssertionError') == [matching]


@pytest.mark.asyncio
async def test_query_by_file_path_returns_tool_and_patch_matches():
  service = InMemoryCompactionService()

  tool_run = ToolRunCompaction(
      event_id='evt-tool-file',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=[],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/target.py', line=12)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-file'),
  )
  patch = PatchCompaction(
      event_id='evt-patch-file',
      compaction_version=1,
      files_changed=['src/target.py'],
      hunks=[
          HunkSummary(
              path='src/target.py',
              anchor_before='a',
              anchor_after='b',
              snippet='-a\n+b',
          )
      ],
      semantic_tags=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-patch-file'),
  )

  await service.save_tool_run_compaction(tool_run)
  await service.save_patch_compaction(patch)

  assert await service.query_by_file_path('src/target.py') == [tool_run, patch]


@pytest.mark.asyncio
async def test_get_observations_filters_by_seq_range():
  service = InMemoryCompactionService()

  first = Observation(
      observation_id='obs-1',
      session_id='session-1',
      start_seq=1,
      end_seq=3,
      text='first',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  second = Observation(
      observation_id='obs-2',
      session_id='session-1',
      start_seq=4,
      end_seq=7,
      text='second',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  third = Observation(
      observation_id='obs-3',
      session_id='session-1',
      start_seq=8,
      end_seq=10,
      text='third',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )

  await service.save_observation(first)
  await service.save_observation(second)
  await service.save_observation(third)

  assert await service.get_observations('session-1', seq_range=(4, 8)) == [
      second
  ]
