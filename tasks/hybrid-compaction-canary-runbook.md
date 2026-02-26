# Hybrid Compaction Production Canary Rollout and Rollback Runbook

## Scope and intent

This runbook covers production rollout and rollback for hybrid compaction in
ADK, specifically the `HybridEventsCompactionConfig` feature flags:

- `enable_deterministic_compaction`
- `enable_observational_memory`
- `enable_hybrid_prompt_assembly`

This is an operations-only runbook for staged canary rollout, kill-switch
activation, rollback, and verification.

## Ownership and escalation

- **Primary owner (Driver):** ADK runtime on-call engineer.
- **Secondary owner (Approver):** Compaction feature owner.
- **Release authority:** Incident commander (IC) or release manager on call.
- **Escalation path:**
  1. Runtime on-call (immediate triage, <=5 minutes).
  2. Compaction feature owner (deep diagnosis, <=15 minutes).
  3. ADK TL/manager (rollback authority if impact is user-visible).
- **Escalation trigger:** Any stop criterion listed in this runbook.

## Preflight checklist

Complete all items before canary traffic is enabled.

- [ ] Confirm target branch and SHA are correct.
- [ ] Confirm rollout owner, approver, and pager coverage are available.
- [ ] Confirm deterministic and hybrid test gates are green at candidate SHA.
- [ ] Confirm rollback target SHA (last known good) is identified and tagged.
- [ ] Confirm kill-switch path is tested in staging and can deploy in <=10
      minutes.
- [ ] Confirm canary cohort definition (sessions, tenants, or traffic percent)
      is documented.

### Preflight command examples (repo-local)

Run from repo root (`/Users/steven/git/adk-python`).

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
```

```bash
uv sync --extra test
```

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/test_config.py \
  tests/unittests/compaction/test_assembly.py \
  tests/unittests/compaction/test_rehydration.py \
  tests/unittests/runners/test_hybrid_compaction.py -q
```

```bash
./scripts/run_hybrid_canary_soak.sh --iterations 30 --stop-on-failure
```

### Canary soak automation script

Use `scripts/run_hybrid_canary_soak.sh` for reproducible local and CI rehearsals.

- **Default test target:**
  `tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration`
- **Default loop count:** `30` runs.
- **Summary output:**
  `CANARY_SOAK_SUMMARY run_count=<n> pass_count=<n> fail_count=<n>`
- **Exit code behavior:**
  - `0` when all executed runs pass.
  - `1` when any run fails.

Example CI/local command:

```bash
./scripts/run_hybrid_canary_soak.sh --iterations 30 --stop-on-failure
```

Recommended promotion threshold:

- **PASS:** `run_count=30`, `pass_count=30`, `fail_count=0`
- **FAIL:** any non-zero `fail_count` or fewer than 30 successful runs.

## Canary progression

Use progressive exposure with explicit go/stop criteria. Do not advance until
all go criteria pass for the current stage.

### Stage 0 - deterministic-only canary (5%)

- **Config:**
  - `enable_deterministic_compaction=True`
  - `enable_observational_memory=False`
  - `enable_hybrid_prompt_assembly=False`
- **Duration:** minimum 60 minutes and at least 100 sessions.
- **Go criteria:**
  - No increase in request failure rate vs baseline.
  - No p95 latency regression >10% vs baseline.
  - No new recurring compaction exceptions in logs.
- **Stop criteria:** any go criterion fails for 10 consecutive minutes.

### Stage 1 - observational memory enabled (10%)

- **Config:**
  - `enable_deterministic_compaction=True`
  - `enable_observational_memory=True`
  - `enable_hybrid_prompt_assembly=False`
- **Duration:** minimum 2 hours and at least 300 sessions.
- **Go criteria:**
  - Stage 0 criteria remain green.
  - No evidence of empty/invalid observations in sampling checks.
  - No elevated retry rate attributable to compaction path.
- **Stop criteria:** user-visible response quality regressions or repeated
  observation-write failures.

### Stage 2 - full hybrid assembly enabled (25% -> 50% -> 100%)

- **Config:**
  - `enable_deterministic_compaction=True`
  - `enable_observational_memory=True`
  - `enable_hybrid_prompt_assembly=True`
- **Progression:** 25% (2 hours) -> 50% (4 hours) -> 100% (after 24-hour soak).
- **Go criteria:**
  - Error budget burn remains within normal range.
  - p95 latency regression <=15% and no sustained timeout spike.
  - Canary cohort quality signals are neutral or improved.
- **Stop criteria:** burn-rate alert, sustained timeout increase, or on-call
  judgment of user impact.

## Observability and alerts

Track these signals continuously during canary windows.

- **Service health:** request error rate, timeout rate, p95/p99 latency.
- **Compaction-path health:** exceptions from runner compaction hook,
  observation writer, reflection writer, and hybrid prompt assembly.
- **Quality proxies:** repeated tool-call loops, repeated identical failure
  signatures, and user-abandon spikes in canary cohort.
- **Alert policy:**
  - Page immediately for Sev2+ user impact or burn-rate alert.
  - Open incident and freeze progression for any stop criterion breach.

### Practical degradation signal checklist

Use this checklist during Stage 0-2 rollouts to confirm degraded-mode signaling
is present and bounded.

- [ ] **Deterministic tool-run compaction degradation signals:**
      `Deterministic tool-run compaction failed for session_id=<...>, invocation_id=<...>, event_id=<...>.`
      - Condition to monitor: repeated exceptions for the same `session_id` or
        bursts across unique `session_id` values.
      - Trigger: >=3 occurrences in 5 minutes for one session, or >=20
        occurrences in 10 minutes fleet-wide.
- [ ] **Deterministic patch compaction degradation signals:**
      `Deterministic patch compaction failed for session_id=<...>, invocation_id=<...>, event_id=<...>.`
      - Condition to monitor: patch-compaction failures tracking with latency or
        timeout increases.
      - Trigger: >=10 occurrences in 10 minutes with any concurrent p95 latency
        regression >10%.
- [ ] **Hybrid assembly fallback signals:**
      `Hybrid prompt assembly failed for session_id=<...>, invocation_id=<...>.`
      - Condition to monitor: fallback frequency and whether affected sessions
        continue to produce successful model responses.
      - Trigger: >=5% of hybrid-enabled invocations over a 15-minute window, or
        any sustained increase over baseline for 30 minutes.
- [ ] **Observational runtime initialization degradation signals (warning):**
      `Skipping observational memory runtime initialization because root agent does not expose canonical_model. app_name=<...>`
      - Condition to monitor: unexpected warnings after enabling
        `enable_observational_memory`.
      - Trigger: warning appears in production canary where the root agent is
        expected to expose `canonical_model`.

Operator notes:

- Treat any one-off signal as degraded-but-tolerated behavior; do not page if
  service-level metrics remain green.
- Treat repeated signals with shared `session_id`/`invocation_id` as a likely
  sticky failure mode that needs mitigation before stage promotion.
- Include `session_id`, `invocation_id`, and `event_id` in incident artifacts
  for targeted replay and root-cause analysis.

### Repo-relevant diagnostic command examples

```bash
./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction.py -q
```

```bash
./.venv/bin/pytest \
  tests/unittests/compaction/storage/test_sqlite_compaction_service.py \
  tests/unittests/compaction/test_observation_writer.py \
  tests/unittests/compaction/test_reflection_writer.py -q
```

## Kill-switch procedure (fast mitigation)

Use this first when there is active user impact and root cause is not yet
confirmed.

1. Set all hybrid compaction switches OFF in the production config source:
   - `enable_deterministic_compaction=False`
   - `enable_observational_memory=False`
   - `enable_hybrid_prompt_assembly=False`
2. Deploy config-only change using normal production deploy pipeline.
3. Confirm new config is active in serving environment.
4. Monitor 15 minutes for recovery in error rate and latency.
5. If recovery is not observed, execute full rollback procedure below.

### Example config shape

```python
from google.adk.compaction.config import HybridEventsCompactionConfig

hybrid_config = HybridEventsCompactionConfig(
    compaction_service=...,  # existing production backend
    compaction_interval=20,
    overlap_size=2,
    enable_deterministic_compaction=False,
    enable_observational_memory=False,
    enable_hybrid_prompt_assembly=False,
)
```

## Rollback procedure (code or config)

Run rollback when kill-switch is insufficient, or when correctness risk remains.

1. Announce rollback in incident channel with current SHA and impact summary.
2. Revert to last known good release SHA.
3. Deploy rollback artifact.
4. Verify production is serving rollback SHA.
5. Keep hybrid switches disabled until postmortem action items are complete.

### Rollback command examples (repo-local preparation)

```bash
git log --oneline -n 20
git checkout <last-known-good-sha>
```

```bash
./.venv/bin/pytest \
  tests/unittests/apps/test_compaction.py \
  tests/unittests/flows/llm_flows/test_compaction_processor.py -q
```

```bash
./.venv/bin/pytest tests/unittests/runners/test_hybrid_compaction_resilience.py -q
```

## Post-rollback verification

Complete all checks before declaring incident mitigated.

- [ ] Error rate returns to baseline band for at least 30 minutes.
- [ ] p95 latency returns to baseline band for at least 30 minutes.
- [ ] No new compaction-path exception bursts in logs.
- [ ] Baseline compaction tests pass at rollback SHA.
- [ ] Incident timeline and final status are posted.

### Verification commands

```bash
./.venv/bin/pytest \
  tests/unittests/apps/test_compaction.py \
  tests/unittests/flows/llm_flows/test_compaction_processor.py \
  tests/unittests/compaction/test_config.py -q
```

## Stop/go summary (single-page criteria)

- **GO:** all stage-specific criteria pass for required soak duration and sample
  size.
- **STOP:** any Sev2+ impact, burn-rate alert, sustained timeout spike,
  compaction exception burst, or quality regression.
- **RESUME after stop:** only after mitigation is active, metrics are stable for
  >=30 minutes, and approver signs off.
- **ROLLBACK mandatory:** if kill-switch does not recover service health within
  15 minutes.
