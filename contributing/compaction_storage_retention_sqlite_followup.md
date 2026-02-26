# Compaction Storage Retention: SQLite Follow-up

This slice introduces retention/eviction API methods on
`BaseCompactionService` and a deterministic implementation in
`InMemoryCompactionService`.

`SqliteCompactionService` is intentionally unchanged in this slice to keep the
change low risk and backward compatible.

## Follow-up Scope for SQLite

- Implement `cleanup_artifacts(...)` with SQL-based deletion for:
  - age-based retention using `created_at`/`updated_at` columns
  - count-based retention per artifact kind (newest-first retention)
  - session-scoped cleanup for observations/reflections/task states
- Implement `evict_session(session_id)` as a targeted delete for
  `observations`, `reflections`, and `task_states` rows.
- Ensure deterministic tie-breaking in count-based eviction by ordering on
  `(created_at, primary_key)`.
- Add unit tests for age/count/session cleanup paths, including mixed data and
  idempotent repeated cleanup calls.
- Verify foreign-key cleanup behavior for index tables tied to
  `tool_run_compactions` and `patch_compactions`.

## Compatibility Expectations

- Existing behavior must remain unchanged unless one of the new cleanup methods
  is explicitly called.
- Existing callers should continue to work because base-class defaults are
  no-op implementations.
