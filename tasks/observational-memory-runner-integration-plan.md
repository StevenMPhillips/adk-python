# Observational Memory Runner Integration Plan (Kickoff)

## Scope of this kickoff slice

This first slice starts runtime integration safely, with strict gating and
no behavior change for existing apps.

- Keep `HybridEventsCompactionConfig.enable_observational_memory=False` as the
  default.
- Initialize observational-memory writer components only when that flag is
  explicitly enabled.
- Keep runtime execution no-op unless an explicit per-invocation trigger is
  provided.

## Trigger points

Planned runtime hook points in `Runner`:

1. End of normal invocation in `Runner.run_async()` after agent events are
   yielded and appended.
2. Optional explicit per-invocation trigger via
   `RunConfig.custom_metadata['trigger_observational_memory_runtime']`.
3. Future trigger sources (deferred): token-window pressure, episode closure,
   repeated failure signatures, and tool-run density thresholds.

Current kickoff behavior:

- On each invocation end, runner checks compaction config.
- If observational memory is enabled, runner lazily initializes runtime writers
  for that session.
- If explicit trigger metadata is absent, runner returns immediately.
- If explicit trigger metadata is present, runner logs and returns (no-op in
  this slice).

## Safety gates

Primary safety gates:

1. **Config gate**: only run when config is
   `HybridEventsCompactionConfig` with deterministic compaction enabled and
   `enable_observational_memory=True`.
2. **Session-local lazy init**: runtime components are cached per session and
   instantiated once per session.
3. **Explicit trigger gate**: even with runtime initialized, execution path is
   inert unless run-level trigger metadata is set.
4. **Backward compatibility**: no changes to default behavior, request format,
   or compaction artifacts for users who do not opt in.

## Failure handling

Failure policy for this slice is degrade-and-continue:

- Runtime initialization exceptions are caught and logged with session context.
- Invocation output is not blocked by observational-memory initialization or
  trigger checks.
- Existing deterministic compaction behavior remains unchanged.

Future failure policy (deferred):

- Per-step retries for writer execution.
- Circuit-breaker and cool-down window for repeated writer failures.
- Optional telemetry counters for trigger attempts, skipped runs, and failures.

## Phased rollout plan

### Phase 0 (this kickoff)

- Add runner plumbing and lazy runtime initialization.
- Add explicit trigger gate with no-op execution path.
- Add tests for config gating and no-op behavior.

### Phase 1

- Implement deterministic data extraction from current invocation events:
  raw turn window, tool-run compactions, patch compactions, and task-state
  lookups.
- Execute `ObservationWriter` on strict trigger conditions only.

### Phase 2

- Integrate `TaskStateUpdater` after successful observation writes.
- Add bounded `ReflectionWriter` trigger cadence.

### Phase 3

- Connect hybrid prompt assembly behavior to observational-memory rollout
  readiness and storage-backed artifacts.
- Add production telemetry, runbook metrics, and canary controls.

## Deferred items (intentionally out of scope now)

- Full writer execution logic and persistence writes.
- Trigger heuristics based on token/runtime signals.
- Live (`run_live`) observational-memory integration.
- New config surface area beyond existing feature flags.
