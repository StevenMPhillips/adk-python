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

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
import sqlite3
import time
from urllib.parse import unquote
from urllib.parse import urlparse

import aiosqlite

from .base_compaction_service import BaseCompactionService
from ..models import Observation
from ..models import PatchCompaction
from ..models import Reflection
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction

PRAGMA_FOREIGN_KEYS = 'PRAGMA foreign_keys = ON'

SCHEMA_VERSION_KEY = 'schema_version'
CURRENT_SCHEMA_VERSION = 1

CREATE_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CREATE_TOOL_RUN_COMPACTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tool_run_compactions (
    event_id TEXT PRIMARY KEY,
    compaction_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

CREATE_PATCH_COMPACTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS patch_compactions (
    event_id TEXT PRIMARY KEY,
    compaction_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

CREATE_OBSERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    observation_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

CREATE_REFLECTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reflections (
    reflection_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    reflection_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

CREATE_TASK_STATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_states (
    session_id TEXT PRIMARY KEY,
    task_state_json TEXT NOT NULL,
    last_updated_seq INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
"""

CREATE_TOOL_RUN_ERROR_SIGNATURES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tool_run_error_signatures (
    event_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    PRIMARY KEY (event_id, signature),
    FOREIGN KEY (event_id)
      REFERENCES tool_run_compactions(event_id)
      ON DELETE CASCADE
);
"""

CREATE_TOOL_RUN_FILE_PATHS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tool_run_file_paths (
    event_id TEXT NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (event_id, path),
    FOREIGN KEY (event_id)
      REFERENCES tool_run_compactions(event_id)
      ON DELETE CASCADE
);
"""

CREATE_PATCH_FILE_PATHS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS patch_file_paths (
    event_id TEXT NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (event_id, path),
    FOREIGN KEY (event_id)
      REFERENCES patch_compactions(event_id)
      ON DELETE CASCADE
);
"""

CREATE_INDEX_SQL = (
    'CREATE INDEX IF NOT EXISTS idx_tool_run_error_signature '
    'ON tool_run_error_signatures(signature);',
    'CREATE INDEX IF NOT EXISTS idx_tool_run_file_path '
    'ON tool_run_file_paths(path);',
    'CREATE INDEX IF NOT EXISTS idx_patch_file_path '
    'ON patch_file_paths(path);',
    'CREATE INDEX IF NOT EXISTS idx_observations_session_seq '
    'ON observations(session_id, start_seq, end_seq);',
    'CREATE INDEX IF NOT EXISTS idx_reflections_session_created '
    'ON reflections(session_id, created_at);',
)

MIGRATIONS = {
    1: (
        CREATE_METADATA_TABLE_SQL,
        CREATE_TOOL_RUN_COMPACTIONS_TABLE_SQL,
        CREATE_PATCH_COMPACTIONS_TABLE_SQL,
        CREATE_OBSERVATIONS_TABLE_SQL,
        CREATE_REFLECTIONS_TABLE_SQL,
        CREATE_TASK_STATES_TABLE_SQL,
        CREATE_TOOL_RUN_ERROR_SIGNATURES_TABLE_SQL,
        CREATE_TOOL_RUN_FILE_PATHS_TABLE_SQL,
        CREATE_PATCH_FILE_PATHS_TABLE_SQL,
        *CREATE_INDEX_SQL,
    ),
}


def _parse_db_path(db_path: str) -> tuple[str, str, bool]:
  """Normalizes a SQLite db path from a URL or filesystem path."""
  if not db_path.startswith(('sqlite:', 'sqlite+aiosqlite:')):
    return db_path, db_path, False

  parsed = urlparse(db_path)
  raw_path = unquote(parsed.path)
  if not raw_path:
    return db_path, db_path, False

  normalized_path = raw_path
  if normalized_path.startswith('//'):
    normalized_path = normalized_path[1:]
  elif normalized_path.startswith('/'):
    normalized_path = normalized_path[1:]

  if parsed.query:
    return normalized_path, f'file:{normalized_path}?{parsed.query}', True

  return normalized_path, normalized_path, False


class SqliteCompactionService(BaseCompactionService):
  """SQLite-backed storage service for compaction artifacts."""

  def __init__(self, db_path: str):
    self._db_path, self._db_connect_path, self._db_connect_uri = _parse_db_path(
        db_path
    )
    self._schema_ready = False
    self._schema_lock = asyncio.Lock()

  async def save_tool_run_compaction(
      self, tool_run_compaction: ToolRunCompaction
  ) -> None:
    await self._ensure_schema()
    file_paths = {ref.path for ref in tool_run_compaction.file_line_refs}

    async with self._get_db_connection() as db:
      now = time.time()
      await db.execute(
          """
          INSERT INTO tool_run_compactions (event_id, compaction_json, created_at)
          VALUES (?, ?, ?)
          ON CONFLICT(event_id) DO UPDATE SET
            compaction_json=excluded.compaction_json,
            created_at=excluded.created_at
          """,
          (
              tool_run_compaction.event_id,
              tool_run_compaction.model_dump_json(exclude_none=True),
              now,
          ),
      )

      await db.execute(
          'DELETE FROM tool_run_error_signatures WHERE event_id=?',
          (tool_run_compaction.event_id,),
      )
      await db.execute(
          'DELETE FROM tool_run_file_paths WHERE event_id=?',
          (tool_run_compaction.event_id,),
      )

      await db.executemany(
          """
          INSERT INTO tool_run_error_signatures (event_id, signature)
          VALUES (?, ?)
          """,
          [
              (tool_run_compaction.event_id, signature)
              for signature in set(tool_run_compaction.error_signatures)
          ],
      )
      await db.executemany(
          """
          INSERT INTO tool_run_file_paths (event_id, path)
          VALUES (?, ?)
          """,
          [(tool_run_compaction.event_id, path) for path in file_paths],
      )
      await db.commit()

  async def save_patch_compaction(
      self, patch_compaction: PatchCompaction
  ) -> None:
    await self._ensure_schema()
    file_paths = set(patch_compaction.files_changed)
    file_paths.update(hunk.path for hunk in patch_compaction.hunks)

    async with self._get_db_connection() as db:
      now = time.time()
      await db.execute(
          """
          INSERT INTO patch_compactions (event_id, compaction_json, created_at)
          VALUES (?, ?, ?)
          ON CONFLICT(event_id) DO UPDATE SET
            compaction_json=excluded.compaction_json,
            created_at=excluded.created_at
          """,
          (
              patch_compaction.event_id,
              patch_compaction.model_dump_json(exclude_none=True),
              now,
          ),
      )

      await db.execute(
          'DELETE FROM patch_file_paths WHERE event_id=?',
          (patch_compaction.event_id,),
      )
      await db.executemany(
          'INSERT INTO patch_file_paths (event_id, path) VALUES (?, ?)',
          [(patch_compaction.event_id, path) for path in file_paths],
      )
      await db.commit()

  async def save_observation(self, observation: Observation) -> None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      await db.execute(
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
          ON CONFLICT(observation_id) DO UPDATE SET
            session_id=excluded.session_id,
            start_seq=excluded.start_seq,
            end_seq=excluded.end_seq,
            observation_json=excluded.observation_json,
            created_at=excluded.created_at
          """,
          (
              observation.observation_id,
              observation.session_id,
              observation.start_seq,
              observation.end_seq,
              observation.model_dump_json(exclude_none=True),
              time.time(),
          ),
      )
      await db.commit()

  async def save_reflection(self, reflection: Reflection) -> None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      await db.execute(
          """
          INSERT INTO reflections (
            reflection_id,
            session_id,
            reflection_json,
            created_at
          )
          VALUES (?, ?, ?, ?)
          ON CONFLICT(reflection_id) DO UPDATE SET
            session_id=excluded.session_id,
            reflection_json=excluded.reflection_json,
            created_at=excluded.created_at
          """,
          (
              reflection.reflection_id,
              reflection.session_id,
              reflection.model_dump_json(exclude_none=True),
              time.time(),
          ),
      )
      await db.commit()

  async def save_task_state(self, task_state: TaskStateAnchor) -> None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      await db.execute(
          """
          INSERT INTO task_states (
            session_id,
            task_state_json,
            last_updated_seq,
            updated_at
          )
          VALUES (?, ?, ?, ?)
          ON CONFLICT(session_id) DO UPDATE SET
            task_state_json=excluded.task_state_json,
            last_updated_seq=excluded.last_updated_seq,
            updated_at=excluded.updated_at
          """,
          (
              task_state.session_id,
              task_state.model_dump_json(exclude_none=True),
              task_state.last_updated_seq,
              time.time(),
          ),
      )
      await db.commit()

  async def get_tool_run_compaction(
      self, event_id: str
  ) -> ToolRunCompaction | None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      async with db.execute(
          'SELECT compaction_json FROM tool_run_compactions WHERE event_id=?',
          (event_id,),
      ) as cursor:
        row = await cursor.fetchone()
    if row is None:
      return None
    return ToolRunCompaction.model_validate_json(row['compaction_json'])

  async def get_patch_compaction(self, event_id: str) -> PatchCompaction | None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      async with db.execute(
          'SELECT compaction_json FROM patch_compactions WHERE event_id=?',
          (event_id,),
      ) as cursor:
        row = await cursor.fetchone()
    if row is None:
      return None
    return PatchCompaction.model_validate_json(row['compaction_json'])

  async def get_observations(
      self,
      session_id: str,
      seq_range: tuple[int, int] | None = None,
  ) -> list[Observation]:
    await self._ensure_schema()
    query = (
        'SELECT observation_json FROM observations WHERE session_id=? '
        'ORDER BY start_seq, end_seq, created_at'
    )
    params: list[object] = [session_id]
    if seq_range is not None:
      start_seq, end_seq = seq_range
      query = (
          'SELECT observation_json FROM observations WHERE session_id=? '
          'AND start_seq >= ? AND end_seq <= ? '
          'ORDER BY start_seq, end_seq, created_at'
      )
      params.extend([start_seq, end_seq])

    async with self._get_db_connection() as db:
      rows = await db.execute_fetchall(query, params)
    return [Observation.model_validate_json(row['observation_json']) for row in rows]

  async def get_latest_reflection(self, session_id: str) -> Reflection | None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      async with db.execute(
          """
          SELECT reflection_json
          FROM reflections
          WHERE session_id=?
          ORDER BY created_at DESC, reflection_id DESC
          LIMIT 1
          """,
          (session_id,),
      ) as cursor:
        row = await cursor.fetchone()
    if row is None:
      return None
    return Reflection.model_validate_json(row['reflection_json'])

  async def get_task_state(self, session_id: str) -> TaskStateAnchor | None:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      async with db.execute(
          'SELECT task_state_json FROM task_states WHERE session_id=?',
          (session_id,),
      ) as cursor:
        row = await cursor.fetchone()
    if row is None:
      return None
    return TaskStateAnchor.model_validate_json(row['task_state_json'])

  async def query_by_error_signature(
      self, signature: str
  ) -> list[ToolRunCompaction]:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      rows = await db.execute_fetchall(
          """
          SELECT tr.compaction_json
          FROM tool_run_compactions tr
          JOIN tool_run_error_signatures es
            ON es.event_id = tr.event_id
          WHERE es.signature = ?
          ORDER BY tr.created_at, tr.event_id
          """,
          (signature,),
      )
    return [
        ToolRunCompaction.model_validate_json(row['compaction_json'])
        for row in rows
    ]

  async def query_by_file_path(
      self, path: str
  ) -> list[ToolRunCompaction | PatchCompaction]:
    await self._ensure_schema()

    async with self._get_db_connection() as db:
      tool_rows = await db.execute_fetchall(
          """
          SELECT tr.compaction_json
          FROM tool_run_compactions tr
          JOIN tool_run_file_paths tfp
            ON tfp.event_id = tr.event_id
          WHERE tfp.path = ?
          ORDER BY tr.created_at, tr.event_id
          """,
          (path,),
      )
      patch_rows = await db.execute_fetchall(
          """
          SELECT pc.compaction_json
          FROM patch_compactions pc
          JOIN patch_file_paths pfp
            ON pfp.event_id = pc.event_id
          WHERE pfp.path = ?
          ORDER BY pc.created_at, pc.event_id
          """,
          (path,),
      )

    tool_results = [
        ToolRunCompaction.model_validate_json(row['compaction_json'])
        for row in tool_rows
    ]
    patch_results = [
        PatchCompaction.model_validate_json(row['compaction_json'])
        for row in patch_rows
    ]
    return [*tool_results, *patch_results]

  @asynccontextmanager
  async def _get_db_connection(self):
    async with aiosqlite.connect(
        self._db_connect_path, uri=self._db_connect_uri
    ) as db:
      db.row_factory = aiosqlite.Row
      await db.execute(PRAGMA_FOREIGN_KEYS)
      yield db

  async def _ensure_schema(self) -> None:
    if self._schema_ready:
      return

    async with self._schema_lock:
      if self._schema_ready:
        return
      async with self._get_db_connection() as db:
        await self._migrate_schema(db)
      self._schema_ready = True

  async def _migrate_schema(self, db: aiosqlite.Connection) -> None:
    schema_version = await self._get_schema_version(db)
    if schema_version >= CURRENT_SCHEMA_VERSION:
      return

    for version in range(schema_version + 1, CURRENT_SCHEMA_VERSION + 1):
      migration_sql = MIGRATIONS.get(version)
      if migration_sql is None:
        raise RuntimeError(f'Missing migration SQL for version {version}.')

      for statement in migration_sql:
        await db.execute(statement)
      await db.execute(
          """
          INSERT INTO metadata (key, value)
          VALUES (?, ?)
          ON CONFLICT(key) DO UPDATE SET value=excluded.value
          """,
          (SCHEMA_VERSION_KEY, str(version)),
      )
    await db.commit()

  async def _get_schema_version(self, db: aiosqlite.Connection) -> int:
    if not await self._metadata_table_exists(db):
      return 0

    async with db.execute(
        'SELECT value FROM metadata WHERE key=?', (SCHEMA_VERSION_KEY,)
    ) as cursor:
      row = await cursor.fetchone()

    if row is None:
      return 0

    try:
      return int(row['value'])
    except ValueError as e:
      raise RuntimeError('Invalid schema version value in metadata table.') from e

  async def _metadata_table_exists(self, db: aiosqlite.Connection) -> bool:
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ) as cursor:
      row = await cursor.fetchone()
    return row is not None

  def is_migration_needed(self) -> bool:
    """Returns whether the database schema is older than current version."""
    if not os.path.exists(self._db_path):
      return False
    try:
      with sqlite3.connect(
          self._db_connect_path, uri=self._db_connect_uri
      ) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        )
        if not cursor.fetchone():
          return True
        cursor.execute(
            'SELECT value FROM metadata WHERE key=?', (SCHEMA_VERSION_KEY,)
        )
        row = cursor.fetchone()
        if not row:
          return True
        return int(row[0]) < CURRENT_SCHEMA_VERSION
    except sqlite3.Error as e:
      raise RuntimeError(
          f'Error accessing database {self._db_path}: {e}'
      ) from e
