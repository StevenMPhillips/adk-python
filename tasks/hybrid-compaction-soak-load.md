# Hybrid Compaction Soak/Load Harness

This task adds a deterministic soak/load harness for hybrid sessions using an
existing hybrid runner test target.

## Script

- Path: `scripts/run_hybrid_soak_load.sh`
- Default iterations: `100`
- Default workers: `1`
- Default test target:
  `tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration`

The harness executes the same deterministic pytest target repeatedly, tracks
pass/fail counts, and emits a single machine-readable summary line:

`HYBRID_SOAK_LOAD_SUMMARY run_count=<N> pass_count=<N> fail_count=<N> duration_seconds=<N>`

## Usage

Quick validation (local):

```bash
scripts/run_hybrid_soak_load.sh --iterations 4 --workers 2
```

Soak profile (serial, stability):

```bash
scripts/run_hybrid_soak_load.sh --iterations 100 --workers 1
```

Load profile (parallel pressure):

```bash
scripts/run_hybrid_soak_load.sh --iterations 100 --workers 4
```

Fail fast on first observed failure:

```bash
scripts/run_hybrid_soak_load.sh --iterations 100 --workers 4 --stop-on-failure
```

## Thresholds

- Quick validation threshold: `fail_count=0` for `--iterations 4 --workers 2`.
- Soak stability threshold: `fail_count=0` for `--iterations 100 --workers 1`.
- Load stability threshold: `fail_count=0` for `--iterations 100 --workers 4`.
- Runtime threshold: no hard gate in this iteration; compare
  `duration_seconds` against your last known-good baseline and investigate
  meaningful regressions.

## Notes

- `--stop-on-failure` stops scheduling new runs after the first failing run is
  observed. Any runs already launched in the current batch still complete.
- Override `--test-target` when needed, but keep targets deterministic and
  based on existing hybrid tests.
