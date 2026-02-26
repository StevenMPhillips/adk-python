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
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import FileLineRef
from google.adk.compaction.models import HunkSummary
from google.adk.compaction.models import Observation
from google.adk.compaction.models import PatchCompaction
from google.adk.compaction.models import Provenance
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.models import ToolRunCompaction
from google.adk.compaction.rehydration import RehydrationExecutor
from google.adk.compaction.rehydration import RehydrationPlanner
from google.adk.compaction.rehydration import RetrievalQuery
from google.adk.compaction.rehydration import TimeRange
from google.adk.compaction.storage.in_memory_compaction_service import InMemoryCompactionService
import pytest


def _sample_stats() -> CompactionStats:
  return CompactionStats(
      raw_tokens_est=120,
      compact_tokens_est=24,
      compression_ratio=5.0,
  )


def _sample_provenance(event_id: str) -> Provenance:
  return Provenance(event_id=event_id)


def _evidence(ref_id: str) -> EvidenceRef:
  return EvidenceRef(ref_type='event', ref_id=ref_id)


def test_rehydration_planner_generates_queries_for_all_triggers():
  planner = RehydrationPlanner()
  task_state = TaskStateAnchor(
      session_id='session-1',
      objective='Fix repeated failing test.',
      constraints=[
          EvidencedItem(
              text=(
                  'Investigate AssertionError in '
                  'tests/test_alpha.py::test_retry at src/foo.py.'
              ),
              evidence_refs=[],
          )
      ],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=30,
  )
  recent_observations = [
      Observation(
          observation_id='obs-1',
          session_id='session-1',
          start_seq=10,
          end_seq=25,
          text='Observed repeated assertion failures.',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[_evidence('evt-1')],
      )
  ]
  recent_compactions = [
      ToolRunCompaction(
          event_id='evt-1',
          compaction_version=1,
          command='pytest tests/test_alpha.py::test_retry',
          exit_code=1,
          error_signatures=['AssertionError'],
          key_errors=['assert x == y'],
          tests_failed=['tests/test_alpha.py::test_retry'],
          file_line_refs=[FileLineRef(path='src/foo.py', line=18)],
          trimmed_trace=['Traceback line'],
          salient_snippets=['assert 1 == 2'],
          stats=_sample_stats(),
          provenance=_sample_provenance('evt-1'),
      ),
      ToolRunCompaction(
          event_id='evt-2',
          compaction_version=1,
          command='pytest tests/test_alpha.py::test_retry',
          exit_code=1,
          error_signatures=['AssertionError'],
          key_errors=['assert x == y'],
          tests_failed=['tests/test_alpha.py::test_retry'],
          file_line_refs=[FileLineRef(path='src/foo.py', line=19)],
          trimmed_trace=['Traceback line 2'],
          salient_snippets=['assert 2 == 3'],
          stats=_sample_stats(),
          provenance=_sample_provenance('evt-2'),
      ),
  ]

  queries = planner.plan(
      task_state=task_state,
      recent_observations=recent_observations,
      recent_compactions=recent_compactions,
      latest_user_message=(
          'Please revisit the earlier src/foo.py bug from previous runs.'
      ),
  )

  assert [query.trigger for query in queries] == [
      'missing_evidence',
      'thrash_detection',
      'user_references_older_specifics',
  ]
  assert queries[0].error_signature == 'AssertionError'
  assert queries[0].test_name == 'tests/test_alpha.py::test_retry'
  assert queries[0].file_path == 'src/foo.py'
  assert queries[0].time_range == TimeRange(start=10, end=25)


@pytest.mark.asyncio
async def test_rehydration_executor_retrieves_and_budgets_evidence_pack():
  service = InMemoryCompactionService()

  tool_in_range = ToolRunCompaction(
      event_id='evt-in',
      compaction_version=1,
      command='pytest tests/test_alpha.py::test_retry',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=['assert x == y'],
      tests_failed=['tests/test_alpha.py::test_retry'],
      file_line_refs=[FileLineRef(path='src/foo.py', line=44)],
      trimmed_trace=['Traceback in range'],
      salient_snippets=['assert in range'],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-in'),
  )
  tool_out_of_range = ToolRunCompaction(
      event_id='evt-out',
      compaction_version=1,
      command='pytest tests/test_alpha.py::test_retry',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=['assert x == y'],
      tests_failed=['tests/test_alpha.py::test_retry'],
      file_line_refs=[FileLineRef(path='src/foo.py', line=50)],
      trimmed_trace=['Traceback out of range'],
      salient_snippets=['assert out of range'],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-out'),
  )
  patch_in_range = PatchCompaction(
      event_id='evt-patch',
      compaction_version=1,
      files_changed=['src/foo.py'],
      hunks=[
          HunkSummary(
              path='src/foo.py',
              anchor_before='old',
              anchor_after='new',
              snippet='-old\n+new',
          )
      ],
      semantic_tags=['bugfix'],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-patch'),
  )

  await service.save_tool_run_compaction(tool_in_range)
  await service.save_tool_run_compaction(tool_out_of_range)
  await service.save_patch_compaction(patch_in_range)

  executor = RehydrationExecutor(service)
  queries = [
      RetrievalQuery(
          trigger='thrash_detection',
          reason='Repeated failures in a loop.',
          error_signature='AssertionError',
          time_range=TimeRange(start=10, end=20),
      ),
      RetrievalQuery(
          trigger='user_references_older_specifics',
          reason='User asked about specific file changes.',
          file_path='src/foo.py',
          time_range=TimeRange(start=10, end=20),
      ),
  ]

  evidence_pack = await executor.execute(
      queries,
      event_time_by_id={
          'evt-in': 15,
          'evt-out': 50,
          'evt-patch': 17,
      },
      raw_excerpt_by_event_id={
          'evt-in': 'Raw excerpt for in-range tool failure.',
      },
  )

  assert evidence_pack.token_budget == 2000
  assert evidence_pack.tokens_est > 0
  assert [
      artifact.event_id for artifact in evidence_pack.compacted_artifacts
  ] == [
      'evt-in',
      'evt-patch',
  ]
  assert evidence_pack.raw_excerpts[0].excerpt == (
      'Raw excerpt for in-range tool failure.'
  )
