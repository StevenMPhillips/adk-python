# Hybrid Compaction Production-Readiness Evidence

Source plan: `tasks/production-readiness-8of10.md`

Execution context:

- Date: 2026-02-25
- Branch: `ralph/lsm-observational-compaction`
- Repo root: `/Users/steven/git/adk-python`

## Gate Results

| Gate | Command run | Result | Evidence |
| --- | --- | --- | --- |
| Prereq | `uv sync --extra test` | PASS | Dependency sync completed with no errors. |
| Gate 1: schema/config contract | `./.venv/bin/pytest tests/unittests/compaction/test_models.py tests/unittests/compaction/test_config.py -q` | PASS | `8 passed` |
| Gate 2: deterministic compactors | `./.venv/bin/pytest tests/unittests/compaction/test_pytest_compactor.py tests/unittests/compaction/test_mypy_compactor.py tests/unittests/compaction/test_ruff_compactor.py tests/unittests/compaction/test_generic_compactor.py tests/unittests/compaction/test_patch_compactor.py tests/unittests/compaction/test_tool_run_compactor_registry.py -q` | PASS | `17 passed` |
| Gate 3: storage/query layer | `./.venv/bin/pytest tests/unittests/compaction/storage/test_in_memory_compaction_service.py tests/unittests/compaction/storage/test_sqlite_compaction_service.py -q` | PASS | `8 passed` |
| Gate 4: observational/task-state behavior | `./.venv/bin/pytest tests/unittests/compaction/test_observation_writer.py tests/unittests/compaction/test_reflection_writer.py tests/unittests/compaction/test_task_state_updater.py -q` | PASS | `13 passed` |
| Gate 5: rehydration + assembly | `./.venv/bin/pytest tests/unittests/compaction/test_rehydration.py tests/unittests/compaction/test_assembly.py -q` | PASS | `4 passed` |
| Gate 6: runner integration | `./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py -q` | PASS | `2 passed` |
| Gate 7: backward compatibility | `./.venv/bin/pytest tests/unittests/apps/test_compaction.py tests/unittests/flows/llm_flows/test_compaction_processor.py -q` | PASS | `34 passed` |
| Gate 8: canary soak reliability | `i=0; for _ in $(seq 1 30); do ./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration -q || break; i=$((i+1)); done; printf 'completed_runs=%s\n' "$i"` | PASS | `completed_runs=30` |

Notes:

- Experimental-feature warnings were emitted in multiple test runs for
  `EventsCompactionConfig` / `HybridEventsCompactionConfig`, but no test
  failures occurred.

## CI-Equivalent Supplementary Checks

These checks were attempted to increase local CI parity.

| Check | Command run | Result | Notes |
| --- | --- | --- | --- |
| Toolchain parity sync | `uv sync --all-extras` | PASS | Installed formatter/import-sort tooling, including `pyink==25.12.0` and `isort==8.0.0`. |
| Formatter availability | `uv run pyink --version` | PASS | `pyink, 25.12.0 (compiled: no)` with CPython 3.12.11. |
| Import-sort availability | `uv run isort --version-number` | PASS | `8.0.0`. |
| Formatter gate (hybrid scope) | `uv run pyink --check --diff --config pyproject.toml src/google/adk/compaction src/google/adk/runners.py src/google/adk/flows/llm_flows/contents.py tests/unittests/compaction tests/unittests/runners/test_hybrid_compaction.py tests/unittests/runners/test_hybrid_compaction_resilience.py` | PASS | Final rerun reported: `44 files would be left unchanged.` |
| Import order gate (hybrid scope) | `uv run isort --check-only src/google/adk/compaction src/google/adk/runners.py src/google/adk/flows/llm_flows/contents.py tests/unittests/compaction tests/unittests/runners/test_hybrid_compaction.py tests/unittests/runners/test_hybrid_compaction_resilience.py` | PASS | Command exited successfully with no import-order violations after scoped isort fixes. |

## Blockers / Gaps

- For the hybrid compaction production-readiness plan itself, all release
  blockers listed in `tasks/production-readiness-8of10.md` are currently clear
  based on executed gates 1-8.
- Style-gate tooling parity blocker is resolved: `pyink` and `isort` are now
  available through the project-standard `uv` workflow and scoped checks pass.

## Confidence Update

- Hybrid compaction production-readiness confidence (plan-scoped): **8/10**.
- Full local CI-parity confidence (including format/lint tools): **8/10** after
  toolchain parity restoration and passing scoped style gates.
