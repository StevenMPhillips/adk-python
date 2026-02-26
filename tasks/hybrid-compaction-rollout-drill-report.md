# Hybrid Compaction Rollout Dry-Run and Rollback Drill Report

## Context

- Branch: `ralph/lsm-observational-compaction`
- Repo: `/Users/steven/git/adk-python`
- Dry-run date: 2026-02-26
- Scope: local rehearsal of canary/soak scripts plus rollback simulation using
  an earlier commit.

## Commands Executed and Outcomes

All commands below were executed from repo root.

| Step | Command | Outcome |
|---|---|---|
| 1 | `git status --short` | PASS (`?? .beads/` only) |
| 2 | `git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD` | PASS (`ralph/lsm-observational-compaction`, initial SHA observed during preflight) |
| 3 | `git log --oneline -n 5` | PASS (used to choose rollback drill target `00ca36e9`) |
| 4 | `./.venv/bin/pytest tests/unittests/compaction/test_config.py tests/unittests/compaction/test_assembly.py tests/unittests/compaction/test_rehydration.py tests/unittests/runners/test_hybrid_compaction.py -q` | PASS (`18 passed, 13 warnings`) |
| 5 | `./scripts/run_hybrid_canary_soak.sh --iterations 30 --stop-on-failure` | PASS (`CANARY_SOAK_SUMMARY run_count=30 pass_count=30 fail_count=0`) |
| 6 | `./scripts/run_hybrid_soak_load.sh --iterations 12 --workers 3 --stop-on-failure` | PASS (`HYBRID_SOAK_LOAD_SUMMARY run_count=12 pass_count=12 fail_count=0 duration_seconds=11`) |
| 7 | `orig_branch="$(git rev-parse --abbrev-ref HEAD)"; orig_sha="$(git rev-parse --short HEAD)"; echo "ORIG_BRANCH=${orig_branch} ORIG_SHA=${orig_sha}"; git checkout --detach 00ca36e9 && ./.venv/bin/pytest tests/unittests/apps/test_compaction.py tests/unittests/flows/llm_flows/test_compaction_processor.py -q; test_status=$?; git checkout "${orig_branch}"; echo "ROLLBACK_DRILL_TEST_EXIT=${test_status}"; exit ${test_status}` | PASS (`34 passed, 23 warnings`; returned to original branch; `ROLLBACK_DRILL_TEST_EXIT=0`) |
| 8 | `./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction_resilience.py -q` | PASS (`7 passed, 7 warnings`) |
| 9 | `git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD` | PASS (`ralph/lsm-observational-compaction`, `68f2f92f`) |

## Rollback Simulation Steps

1. Identified rollback candidate from recent history with `git log --oneline -n 5`.
2. Captured current branch/SHA in-shell (`orig_branch`, `orig_sha`).
3. Checked out rollback target in detached mode: `git checkout --detach 00ca36e9`.
4. Ran rollback verification tests at rollback target:
   - `tests/unittests/apps/test_compaction.py`
   - `tests/unittests/flows/llm_flows/test_compaction_processor.py`
5. Returned to original branch via `git checkout "${orig_branch}"`.
6. Confirmed drill test exit code `0` and reran resilience suite on current
   branch.

## Findings

- Canary soak gate command completed cleanly at the documented threshold (30/30
  passes, no failures).
- Soak/load harness behaved deterministically in quick parallel profile
  (12 runs, 3 workers, zero failures).
- Rollback drill workflow succeeded with detached checkout and branch restore.
- Test output consistently includes experimental-feature warnings for compaction
  config classes; no functional failures observed.

## Action Items

- Clarify runbook with explicit local drill commands for canary + soak/load that
  are faster than release thresholds.
- Clarify rollback runbook to capture original branch before detached checkout
  and verify branch restoration after drill.
