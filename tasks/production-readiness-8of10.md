# Hybrid Compaction: 8/10 Production Readiness Plan

## Target

Reach **8/10 production confidence** for hybrid compaction by proving four
things with repeatable evidence:

1. deterministic artifact extraction is reliable,
2. hybrid assembly/rehydration behaves correctly,
3. backward compatibility with existing compaction is intact,
4. rollout can be stopped safely using explicit blockers.

This plan is scoped to the current implementation under
`src/google/adk/compaction/`, runner hooks in `src/google/adk/runners.py`, and
hybrid prompt assembly in `src/google/adk/flows/llm_flows/contents.py`.

## Measurable Gates (with exact verification commands)

Run from repo root. Use the test venv directly.

Prereq:

```bash
uv sync --extra test
```

### Gate 1 - Schema/config contract is stable

- **Threshold:** 100% pass for model serialization and config defaults.
- **Command:**

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/test_models.py \
  tests/unittests/compaction/test_config.py -q
```

### Gate 2 - Deterministic compactors preserve needles and compress hard

- **Threshold:** 100% pass for pytest/mypy/ruff/generic/patch compactors.
- **Threshold:** pytest compactor keeps built-in assertion
  `compression_ratio > 10.0` green.
- **Command:**

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/test_pytest_compactor.py \
  tests/unittests/compaction/test_mypy_compactor.py \
  tests/unittests/compaction/test_ruff_compactor.py \
  tests/unittests/compaction/test_generic_compactor.py \
  tests/unittests/compaction/test_patch_compactor.py \
  tests/unittests/compaction/test_tool_run_compactor_registry.py -q
```

### Gate 3 - Storage/query layer is reliable (memory + sqlite)

- **Threshold:** 100% pass for save/get/query behavior in both storage backends.
- **Threshold:** `query_by_error_signature` and `query_by_file_path` must pass
  in both implementations.
- **Command:**

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/storage/test_in_memory_compaction_service.py \
  tests/unittests/compaction/storage/test_sqlite_compaction_service.py -q
```

### Gate 4 - Observational memory and task-state behavior is sound

- **Threshold:** 100% pass for observation/reflection writers and task-state
  provenance/versioning behavior.
- **Command:**

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/test_observation_writer.py \
  tests/unittests/compaction/test_reflection_writer.py \
  tests/unittests/compaction/test_task_state_updater.py -q
```

### Gate 5 - Rehydration and hybrid prompt assembly work end-to-end

- **Threshold:** 100% pass for rehydration planner/executor and hybrid assembly.
- **Threshold:** evidence-pack token budgeting assertions stay green.
- **Command:**

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/test_rehydration.py \
  tests/unittests/compaction/test_assembly.py -q
```

### Gate 6 - Runner integration works for multi-turn sessions

- **Threshold:** 100% pass for hybrid runner tests.
- **Threshold:** multi-turn integration assertions remain green, including:
  - deterministic artifact creation for tool responses,
  - observation/task-state integration,
  - `hybrid_tokens_est < llm_only_tokens_est`.
- **Command:**

```bash
./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py -q
```

### Gate 7 - Backward compatibility is preserved

- **Threshold:** 100% pass for existing (non-hybrid) compaction paths.
- **Command:**

```bash
./.venv/bin/pytest \
  tests/unittests/apps/test_compaction.py \
  tests/unittests/flows/llm_flows/test_compaction_processor.py -q
```

### Gate 8 - Canary soak reliability before production

- **Threshold:** 30 consecutive clean runs of the long-session hybrid
  integration test (0 failures).
- **Threshold:** no intermittent failures from deterministic compaction,
  observation writing, or rehydration assertions.
- **Command:**

```bash
for i in $(seq 1 30); do
  ./.venv/bin/pytest \
    tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q || break
done
```

## Staged Rollout Criteria

## Stage A - Dev Complete (feature ready for canary)

Entry:

- hybrid compaction code merged behind config flags:
  - `enable_deterministic_compaction=True`
  - `enable_observational_memory=False` by default
  - `enable_hybrid_prompt_assembly=False` by default

Exit (must all be true):

- Gates 1-7 pass on current branch/commit.
- No new failures in compaction-adjacent unit tests.
- Blocking criteria (below) all clear.

Verification command bundle:

```bash
./.venv/bin/pytest tests/unittests/compaction tests/unittests/runners/test_hybrid_compaction.py tests/unittests/apps/test_compaction.py tests/unittests/flows/llm_flows/test_compaction_processor.py -q
```

## Stage B - Canary (limited production exposure)

Canary scope:

- enable hybrid compaction for one internal coding-agent app/session cohort,
  with deterministic compaction on and observational memory/assembly enabled
  only after deterministic-only canary is stable.

Exit (must all be true):

- Gate 8 passes (30 clean long-session replays).
- No blocker triggered in 48h canary window.
- If canary includes SQLite persistence, storage tests are green at canary SHA.

Verification command bundle:

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/storage/test_sqlite_compaction_service.py \
  tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q
for i in $(seq 1 30); do
  ./.venv/bin/pytest \
    tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q || break
done
```

Rollback trigger during canary:

- disable hybrid path immediately by setting
  `enable_deterministic_compaction=False` and
  `enable_hybrid_prompt_assembly=False`.

## Stage C - Full Production

Entry:

- Stage B completed with no blockers.

Exit (must all be true):

- Full gate suite (1-8) remains green at release candidate SHA.
- Backward compatibility gate still green after release branch cut.
- Rollback path is validated and documented in release notes.

Release verification command bundle:

```bash
./.venv/bin/pytest tests/unittests/compaction tests/unittests/runners/test_hybrid_compaction.py tests/unittests/apps/test_compaction.py tests/unittests/flows/llm_flows/test_compaction_processor.py -q
for i in $(seq 1 30); do
  ./.venv/bin/pytest \
    tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q || break
done
```

## Risk Register and Blocking Criteria

| Risk | Signal / measurable blocker | Verification command | Block? |
| --- | --- | --- | --- |
| Deterministic compactor drops critical evidence (error signature, file:line, test id) | Any failure in compactor tests (especially pytest ratio/evidence checks) | `./.venv/bin/pytest tests/unittests/compaction/test_pytest_compactor.py tests/unittests/compaction/test_mypy_compactor.py tests/unittests/compaction/test_ruff_compactor.py tests/unittests/compaction/test_generic_compactor.py tests/unittests/compaction/test_patch_compactor.py -q` | Yes |
| Runner hook misses artifacts for tool responses | Any failure in runner hybrid integration tests | `./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py -q` | Yes |
| Rehydration retrieves wrong/empty evidence under thrash | Any failure in rehydration + long-session hybrid test | `./.venv/bin/pytest tests/unittests/compaction/test_rehydration.py tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q` | Yes |
| Prompt assembly regresses token behavior | Loss of assertion `hybrid_tokens_est < llm_only_tokens_est` in runner integration | `./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q` | Yes |
| SQLite persistence/query drift | Any failure in sqlite compaction storage tests | `./.venv/bin/pytest tests/unittests/compaction/storage/test_sqlite_compaction_service.py -q` | Yes |
| Legacy compaction behavior regresses | Any failure in existing compaction processor/app tests | `./.venv/bin/pytest tests/unittests/apps/test_compaction.py tests/unittests/flows/llm_flows/test_compaction_processor.py -q` | Yes |
| Flaky behavior appears under repeated long sessions | Fewer than 30/30 green canary-soak runs | `for i in $(seq 1 30); do ./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q || break; done` | Yes |

Blocking rule:

- **Any single blocker above is release-blocking** for canary promotion or full
  production rollout.
- Confidence is capped at **6/10** until all blockers are clear.
- Confidence can be marked **8/10** only when Gates 1-8 are all green.
