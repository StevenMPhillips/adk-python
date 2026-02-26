# Production Readiness 9/10 Hardening Evidence

Date: 2026-02-26
Branch: `ralph/lsm-observational-compaction`
Bead: `adk-python-43i.2`

## Scope and intent

This report captures CI-like hardening evidence for hybrid observational
compaction by running:

- full `tests/unittests/compaction` suite,
- hybrid runner integration/resilience suites,
- broader non-hybrid sanity checks in apps/flows compaction paths, and
- style checks (`pyink` and `isort`) on hybrid test scope.

Only executed commands are reported.

## Executed commands and results

1. ` .venv/bin/pytest tests/unittests/compaction`
   - Result: PASS
   - Collected: 69 tests
   - Outcome: `69 passed, 1 warning in 2.05s`
   - Note: warning is experimental-feature warning for `EventsCompactionConfig`.

2. ` .venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py tests/unittests/runners/test_hybrid_compaction_resilience.py`
   - Result: PASS
   - Collected: 5 tests
   - Outcome: `5 passed, 5 warnings in 1.92s`
   - Note: warnings are experimental-feature warnings in hybrid and config tests.

3. ` .venv/bin/pytest tests/unittests/apps/test_compaction.py tests/unittests/flows/llm_flows/test_compaction_processor.py`
   - Result: PASS
   - Collected: 34 tests
   - Outcome: `34 passed, 23 warnings in 1.54s`
   - Note: warnings are experimental-feature warnings for compaction config paths.

4. ` .venv/bin/pyink --check --diff --config pyproject.toml tests/unittests/compaction/test_hybrid_compaction_perf.py tests/unittests/runners/test_hybrid_compaction.py tests/unittests/runners/test_hybrid_compaction_resilience.py`
   - Result: PASS
   - Outcome: `3 files would be left unchanged`

5. ` .venv/bin/isort --check-only tests/unittests/compaction/test_hybrid_compaction_perf.py tests/unittests/runners/test_hybrid_compaction.py tests/unittests/runners/test_hybrid_compaction_resilience.py`
   - Result: PASS
   - Outcome: no output, zero non-zero exit indicators

## Evidence summary

- Test commands executed: 3
- Lint/format check commands executed: 2
- Total commands executed for this report: 5
- Failed commands: 0
- Net test outcomes: `108 passed` across executed test suites

## Residual risks

1. Experimental-status warning remains active for `EventsCompactionConfig` and
   `HybridEventsCompactionConfig`; behavior/contract may still evolve.
2. This evidence is unit-test heavy; no fresh integration/e2e or load/soak
   evidence is included in this run.
3. Platform matrix coverage is not expanded here (single local environment).
4. `isort --check-only` passed silently; result is inferred from successful
   command completion and lack of non-zero exit output.

## Recommendation

Recommendation: **pilot-only candidate**.

Rationale: hybrid + non-hybrid compaction paths and style gates pass cleanly in
this run, but experimental warnings and absent broader environment/perf matrix
evidence leave residual risk for immediate full-production promotion.
