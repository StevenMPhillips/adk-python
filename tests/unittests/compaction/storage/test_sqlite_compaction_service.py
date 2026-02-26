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

import logging
import sqlite3

import aiosqlite
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
from google.adk.compaction.storage.sqlite_compaction_service import SCHEMA_VERSION_KEY
from google.adk.compaction.storage.sqlite_compaction_service import SqliteCompactionService
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


@pytest.fixture
async def sqlite_service_factory():
  services: list[SqliteCompactionService] = []

  def _factory(db_path) -> SqliteCompactionService:
    service = SqliteCompactionService(str(db_path))
    services.append(service)
    return service

  yield _factory

  for service in services:
    await service.close()


@pytest.mark.asyncio
async def test_sqlite_service_save_and_get_all_artifact_types(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

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
      current_plan=[_sample_item('Implement sqlite service.', 'evt-patch-1')],
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
async def test_sqlite_service_queries_are_indexed_and_return_matches(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  matching_tool = ToolRunCompaction(
      event_id='evt-tool-file',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/target.py', line=12)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-file'),
  )
  non_matching_tool = ToolRunCompaction(
      event_id='evt-tool-other',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['TypeError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/other.py', line=2)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-other'),
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

  await service.save_tool_run_compaction(matching_tool)
  await service.save_tool_run_compaction(non_matching_tool)
  await service.save_patch_compaction(patch)

  assert await service.query_by_error_signature('AssertionError') == [
      matching_tool
  ]
  assert await service.query_by_file_path('src/target.py') == [
      matching_tool,
      patch,
  ]

  with sqlite3.connect(db_path) as conn:
    index_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
  index_names = {row[0] for row in index_rows}
  assert 'idx_tool_run_error_signature' in index_names
  assert 'idx_tool_run_file_path' in index_names
  assert 'idx_patch_file_path' in index_names


@pytest.mark.asyncio
async def test_sqlite_service_observations_filter_by_seq_range(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

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


@pytest.mark.asyncio
async def test_sqlite_service_migrates_from_metadata_version_zero(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  with sqlite3.connect(db_path) as conn:
    conn.execute(
        'CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
    )
    conn.execute(
        'INSERT INTO metadata (key, value) VALUES (?, ?)',
        (SCHEMA_VERSION_KEY, '0'),
    )
    conn.commit()

  service = sqlite_service_factory(db_path)

  tool_run = ToolRunCompaction(
      event_id='evt-tool-1',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/target.py', line=1)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-1'),
  )
  await service.save_tool_run_compaction(tool_run)

  with sqlite3.connect(db_path) as conn:
    version = conn.execute(
        'SELECT value FROM metadata WHERE key=?', (SCHEMA_VERSION_KEY,)
    ).fetchone()[0]
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

  table_names = {row[0] for row in table_rows}
  assert version == '1'
  assert 'tool_run_compactions' in table_names
  assert 'patch_compactions' in table_names
  assert 'observations' in table_names
  assert 'reflections' in table_names
  assert 'task_states' in table_names


@pytest.mark.asyncio
async def test_save_tool_run_compaction_rolls_back_on_related_insert_failure(
    tmp_path, monkeypatch, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  original = ToolRunCompaction(
      event_id='evt-tool-rollback',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/original.py', line=1)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-rollback'),
  )
  await service.save_tool_run_compaction(original)

  updated = ToolRunCompaction(
      event_id='evt-tool-rollback',
      compaction_version=1,
      command='pytest tests/unittests',
      exit_code=2,
      error_signatures=['ValueError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/updated.py', line=2)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-rollback'),
  )

  db = await service._get_or_create_connection()
  original_executemany = db.executemany

  async def _failing_executemany(sql: str, params):
    if 'tool_run_file_paths' in sql:
      raise RuntimeError('forced tool_run_file_paths failure')
    return await original_executemany(sql, params)

  monkeypatch.setattr(db, 'executemany', _failing_executemany)

  with pytest.raises(RuntimeError, match='forced tool_run_file_paths failure'):
    await service.save_tool_run_compaction(updated)

  assert await service.get_tool_run_compaction('evt-tool-rollback') == original
  assert await service.query_by_error_signature('AssertionError') == [original]


@pytest.mark.asyncio
async def test_save_patch_compaction_rolls_back_on_path_insert_failure(
    tmp_path, monkeypatch, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  original = PatchCompaction(
      event_id='evt-patch-rollback',
      compaction_version=1,
      files_changed=['src/original.py'],
      hunks=[],
      semantic_tags=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-patch-rollback'),
  )
  await service.save_patch_compaction(original)

  updated = PatchCompaction(
      event_id='evt-patch-rollback',
      compaction_version=1,
      files_changed=['src/updated.py'],
      hunks=[],
      semantic_tags=['refactor'],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-patch-rollback'),
  )

  db = await service._get_or_create_connection()
  original_executemany = db.executemany

  async def _failing_executemany(sql: str, params):
    if 'patch_file_paths' in sql:
      raise RuntimeError('forced patch_file_paths failure')
    return await original_executemany(sql, params)

  monkeypatch.setattr(db, 'executemany', _failing_executemany)

  with pytest.raises(RuntimeError, match='forced patch_file_paths failure'):
    await service.save_patch_compaction(updated)

  assert await service.get_patch_compaction('evt-patch-rollback') == original
  assert await service.query_by_file_path('src/original.py') == [original]


@pytest.mark.asyncio
async def test_single_row_getter_raises_clear_error_for_corrupt_json(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  tool_run = ToolRunCompaction(
      event_id='evt-corrupt-single',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=[],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-corrupt-single'),
  )
  await service.save_tool_run_compaction(tool_run)

  with sqlite3.connect(db_path) as conn:
    conn.execute(
        'UPDATE tool_run_compactions SET compaction_json=? WHERE event_id=?',
        ('{not-valid-json', 'evt-corrupt-single'),
    )
    conn.commit()

  with pytest.raises(RuntimeError, match='evt-corrupt-single'):
    await service.get_tool_run_compaction('evt-corrupt-single')


@pytest.mark.asyncio
async def test_multi_row_queries_skip_corrupt_json_and_log_warning(
    tmp_path, caplog, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  observation = Observation(
      observation_id='obs-valid',
      session_id='session-1',
      start_seq=1,
      end_seq=1,
      text='valid observation',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  await service.save_observation(observation)

  tool_run = ToolRunCompaction(
      event_id='evt-corrupt-query',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/query.py', line=3)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-corrupt-query'),
  )
  patch = PatchCompaction(
      event_id='evt-valid-patch',
      compaction_version=1,
      files_changed=['src/query.py'],
      hunks=[],
      semantic_tags=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-valid-patch'),
  )
  await service.save_tool_run_compaction(tool_run)
  await service.save_patch_compaction(patch)

  with sqlite3.connect(db_path) as conn:
    conn.execute(
        """
        INSERT INTO observations (
          observation_id,
          session_id,
          start_seq,
          end_seq,
          observation_json,
          created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            'obs-corrupt',
            'session-1',
            2,
            2,
            '{bad-json',
            0.0,
        ),
    )
    conn.execute(
        'UPDATE tool_run_compactions SET compaction_json=? WHERE event_id=?',
        ('{bad-json', 'evt-corrupt-query'),
    )
    conn.commit()

  caplog.set_level(logging.WARNING)
  assert await service.get_observations('session-1') == [observation]
  assert await service.query_by_file_path('src/query.py') == [patch]
  assert 'Skipping invalid observation JSON row in sqlite store.' in caplog.text
  assert (
      'Skipping invalid tool_run_compaction JSON row for record' in caplog.text
  )


@pytest.mark.asyncio
async def test_sqlite_service_reuses_single_aiosqlite_connection(
    tmp_path, monkeypatch, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  connect_call_count = 0
  original_connect = aiosqlite.connect

  async def _counting_connect(*args, **kwargs):
    nonlocal connect_call_count
    connect_call_count += 1
    return await original_connect(*args, **kwargs)

  monkeypatch.setattr(
      'google.adk.compaction.storage.sqlite_compaction_service.aiosqlite.connect',
      _counting_connect,
  )
  service = sqlite_service_factory(db_path)

  tool_run = ToolRunCompaction(
      event_id='evt-reuse-1',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/reuse.py', line=1)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-reuse-1'),
  )

  await service.save_tool_run_compaction(tool_run)
  await service.get_tool_run_compaction('evt-reuse-1')
  await service.query_by_error_signature('AssertionError')

  assert connect_call_count == 1


@pytest.mark.asyncio
async def test_is_migration_needed_async_matches_sync_api(
    tmp_path, sqlite_service_factory
):
  missing_db = sqlite_service_factory(tmp_path / 'missing.db')
  assert missing_db.is_migration_needed() is False
  assert await missing_db.is_migration_needed_async() is False

  db_path = tmp_path / 'compactions.db'
  with sqlite3.connect(db_path) as conn:
    conn.execute(
        'CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
    )
    conn.execute(
        'INSERT INTO metadata (key, value) VALUES (?, ?)',
        (SCHEMA_VERSION_KEY, '0'),
    )
    conn.commit()

  service = sqlite_service_factory(db_path)
  assert service.is_migration_needed() is True
  assert await service.is_migration_needed_async() is True


@pytest.mark.asyncio
async def test_cleanup_artifacts_evicts_records_by_age_and_cascades_indexes(
    tmp_path, monkeypatch, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  timestamps = iter([
      100.0,
      111.0,
      101.0,
      112.0,
      102.0,
      113.0,
      114.0,
      103.0,
      115.0,
      104.0,
      116.0,
  ])
  monkeypatch.setattr(
      'google.adk.compaction.storage.sqlite_compaction_service.time.time',
      lambda: next(timestamps),
  )

  old_tool_run = ToolRunCompaction(
      event_id='evt-old-tool',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['OldError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/old.py', line=1)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-old-tool'),
  )
  new_tool_run = ToolRunCompaction(
      event_id='evt-new-tool',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['NewError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[FileLineRef(path='src/new.py', line=1)],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-new-tool'),
  )
  old_patch = PatchCompaction(
      event_id='evt-old-patch',
      compaction_version=1,
      files_changed=['src/old.py'],
      hunks=[],
      semantic_tags=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-old-patch'),
  )
  new_patch = PatchCompaction(
      event_id='evt-new-patch',
      compaction_version=1,
      files_changed=['src/new.py'],
      hunks=[],
      semantic_tags=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-new-patch'),
  )
  old_observation = Observation(
      observation_id='obs-old',
      session_id='session-a',
      start_seq=1,
      end_seq=1,
      text='old',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  new_observation = Observation(
      observation_id='obs-new',
      session_id='session-a',
      start_seq=2,
      end_seq=2,
      text='new',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  other_session_observation = Observation(
      observation_id='obs-other',
      session_id='session-b',
      start_seq=1,
      end_seq=1,
      text='other',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  old_reflection = Reflection(
      reflection_id='refl-old',
      session_id='session-a',
      covers_observation_ids=['obs-old'],
      text='old',
      stable_facts=[],
      recurring_failures=[],
      strategy_updates=[],
      evidence_refs=[],
  )
  new_reflection = Reflection(
      reflection_id='refl-new',
      session_id='session-a',
      covers_observation_ids=['obs-new'],
      text='new',
      stable_facts=[],
      recurring_failures=[],
      strategy_updates=[],
      evidence_refs=[],
  )
  old_task_state = TaskStateAnchor(
      session_id='session-a',
      objective='old',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=1,
  )
  new_task_state = TaskStateAnchor(
      session_id='session-b',
      objective='new',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=2,
  )

  await service.save_tool_run_compaction(old_tool_run)
  await service.save_tool_run_compaction(new_tool_run)
  await service.save_patch_compaction(old_patch)
  await service.save_patch_compaction(new_patch)
  await service.save_observation(old_observation)
  await service.save_observation(new_observation)
  await service.save_observation(other_session_observation)
  await service.save_reflection(old_reflection)
  await service.save_reflection(new_reflection)
  await service.save_task_state(old_task_state)
  await service.save_task_state(new_task_state)

  cleanup_stats = await service.cleanup_artifacts(max_age_seconds=8, now=116.0)

  assert cleanup_stats.tool_run_compactions_deleted == 1
  assert cleanup_stats.patch_compactions_deleted == 1
  assert cleanup_stats.observations_deleted == 1
  assert cleanup_stats.reflections_deleted == 1
  assert cleanup_stats.task_states_deleted == 1
  assert cleanup_stats.total_deleted == 5
  assert await service.get_tool_run_compaction('evt-old-tool') is None
  assert await service.get_tool_run_compaction('evt-new-tool') == new_tool_run
  assert await service.get_patch_compaction('evt-old-patch') is None
  assert await service.get_patch_compaction('evt-new-patch') == new_patch
  assert await service.get_observations('session-a') == [new_observation]
  assert await service.get_observations('session-b') == [
      other_session_observation
  ]
  assert await service.get_latest_reflection('session-a') == new_reflection
  assert await service.get_task_state('session-a') is None
  assert await service.get_task_state('session-b') == new_task_state

  with sqlite3.connect(db_path) as conn:
    old_tool_signature_rows = conn.execute(
        'SELECT COUNT(*) FROM tool_run_error_signatures WHERE event_id=?',
        ('evt-old-tool',),
    ).fetchone()[0]
    old_tool_path_rows = conn.execute(
        'SELECT COUNT(*) FROM tool_run_file_paths WHERE event_id=?',
        ('evt-old-tool',),
    ).fetchone()[0]
    old_patch_path_rows = conn.execute(
        'SELECT COUNT(*) FROM patch_file_paths WHERE event_id=?',
        ('evt-old-patch',),
    ).fetchone()[0]

  assert old_tool_signature_rows == 0
  assert old_tool_path_rows == 0
  assert old_patch_path_rows == 0


@pytest.mark.asyncio
async def test_cleanup_artifacts_scoped_by_session_respects_count_policy(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  for observation_id in ('obs-a-1', 'obs-a-2', 'obs-a-3'):
    await service.save_observation(
        Observation(
            observation_id=observation_id,
            session_id='session-a',
            start_seq=1,
            end_seq=1,
            text=observation_id,
            decisions=[],
            learned_constraints=[],
            open_questions=[],
            next_steps=[],
            evidence_refs=[],
        )
    )
  await service.save_observation(
      Observation(
          observation_id='obs-b-1',
          session_id='session-b',
          start_seq=1,
          end_seq=1,
          text='obs-b-1',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[],
      )
  )

  for reflection_id in ('refl-a-1', 'refl-a-2', 'refl-a-3'):
    await service.save_reflection(
        Reflection(
            reflection_id=reflection_id,
            session_id='session-a',
            covers_observation_ids=[],
            text=reflection_id,
            stable_facts=[],
            recurring_failures=[],
            strategy_updates=[],
            evidence_refs=[],
        )
    )
  await service.save_reflection(
      Reflection(
          reflection_id='refl-b-1',
          session_id='session-b',
          covers_observation_ids=[],
          text='refl-b-1',
          stable_facts=[],
          recurring_failures=[],
          strategy_updates=[],
          evidence_refs=[],
      )
  )

  task_state_a = TaskStateAnchor(
      session_id='session-a',
      objective='A',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=1,
  )
  task_state_b = TaskStateAnchor(
      session_id='session-b',
      objective='B',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=1,
  )
  tool_run = ToolRunCompaction(
      event_id='evt-tool-scoped-cleanup',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['ScopedError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-scoped-cleanup'),
  )

  await service.save_task_state(task_state_a)
  await service.save_task_state(task_state_b)
  await service.save_tool_run_compaction(tool_run)

  cleanup_stats = await service.cleanup_artifacts(
      max_records_per_kind=1, session_id='session-a'
  )

  assert cleanup_stats.tool_run_compactions_deleted == 0
  assert cleanup_stats.patch_compactions_deleted == 0
  assert cleanup_stats.observations_deleted == 2
  assert cleanup_stats.reflections_deleted == 2
  assert cleanup_stats.task_states_deleted == 0
  assert await service.get_observations('session-a') == [
      Observation(
          observation_id='obs-a-3',
          session_id='session-a',
          start_seq=1,
          end_seq=1,
          text='obs-a-3',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[],
      )
  ]
  assert await service.get_observations('session-b') == [
      Observation(
          observation_id='obs-b-1',
          session_id='session-b',
          start_seq=1,
          end_seq=1,
          text='obs-b-1',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[],
      )
  ]
  latest_reflection_a = await service.get_latest_reflection('session-a')
  latest_reflection_b = await service.get_latest_reflection('session-b')
  assert latest_reflection_a is not None
  assert latest_reflection_b is not None
  assert latest_reflection_a.reflection_id == 'refl-a-3'
  assert latest_reflection_b.reflection_id == 'refl-b-1'
  assert await service.get_task_state('session-a') == task_state_a
  assert await service.get_task_state('session-b') == task_state_b
  assert await service.get_tool_run_compaction('evt-tool-scoped-cleanup') == (
      tool_run
  )


@pytest.mark.asyncio
async def test_evict_session_removes_only_target_session_scoped_records(
    tmp_path, sqlite_service_factory
):
  db_path = tmp_path / 'compactions.db'
  service = sqlite_service_factory(db_path)

  session_a_observation = Observation(
      observation_id='obs-a',
      session_id='session-a',
      start_seq=1,
      end_seq=1,
      text='session-a',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  session_b_observation = Observation(
      observation_id='obs-b',
      session_id='session-b',
      start_seq=1,
      end_seq=1,
      text='session-b',
      decisions=[],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[],
  )
  session_a_reflection = Reflection(
      reflection_id='refl-a',
      session_id='session-a',
      covers_observation_ids=['obs-a'],
      text='session-a reflection',
      stable_facts=[],
      recurring_failures=[],
      strategy_updates=[],
      evidence_refs=[],
  )
  session_b_reflection = Reflection(
      reflection_id='refl-b',
      session_id='session-b',
      covers_observation_ids=['obs-b'],
      text='session-b reflection',
      stable_facts=[],
      recurring_failures=[],
      strategy_updates=[],
      evidence_refs=[],
  )
  session_a_task_state = TaskStateAnchor(
      session_id='session-a',
      objective='A',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=1,
  )
  session_b_task_state = TaskStateAnchor(
      session_id='session-b',
      objective='B',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=1,
  )
  tool_run = ToolRunCompaction(
      event_id='evt-tool-evict',
      compaction_version=1,
      command='pytest',
      exit_code=1,
      error_signatures=['EvictError'],
      key_errors=[],
      tests_failed=[],
      file_line_refs=[],
      trimmed_trace=[],
      salient_snippets=[],
      stats=_sample_stats(),
      provenance=_sample_provenance('evt-tool-evict'),
  )

  await service.save_observation(session_a_observation)
  await service.save_observation(session_b_observation)
  await service.save_reflection(session_a_reflection)
  await service.save_reflection(session_b_reflection)
  await service.save_task_state(session_a_task_state)
  await service.save_task_state(session_b_task_state)
  await service.save_tool_run_compaction(tool_run)

  cleanup_stats = await service.evict_session('session-a')

  assert cleanup_stats.observations_deleted == 1
  assert cleanup_stats.reflections_deleted == 1
  assert cleanup_stats.task_states_deleted == 1
  assert await service.get_observations('session-a') == []
  assert await service.get_latest_reflection('session-a') is None
  assert await service.get_task_state('session-a') is None
  assert await service.get_observations('session-b') == [session_b_observation]
  assert (
      await service.get_latest_reflection('session-b') == session_b_reflection
  )
  assert await service.get_task_state('session-b') == session_b_task_state
  assert await service.get_tool_run_compaction('evt-tool-evict') == tool_run
