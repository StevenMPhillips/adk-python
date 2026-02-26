# LongMemEval Harness Plan for ADK Hybrid Memory

## Goal

Build a reproducible benchmark harness that evaluates this branch's hybrid
compaction and observational-memory behavior on LongMemEval.

Primary outcome:
- Run LongMemEval end-to-end against ADK agent configurations and produce a
  versioned results report.

## Why LongMemEval Fits

LongMemEval is a public benchmark for long-term interactive memory in chat
assistants. It targets capabilities that match this work:
- information extraction across long histories
- multi-session reasoning
- temporal reasoning
- knowledge updates
- abstention

It is suitable for production-readiness evidence because it stresses sustained
chat memory behavior, not only short-context QA.

## Benchmark Sources

- Paper: https://arxiv.org/abs/2410.10813
- Code: https://github.com/xiaowu0162/LongMemEval
- Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## Scope

### In Scope (Phase 1)

- QA correctness on `longmemeval_oracle.json` and `longmemeval_s_cleaned.json`
- comparison across ADK modes:
  1) baseline memory config
  2) hybrid compaction enabled
  3) hybrid + observational memory enabled
- reproducible output artifacts (`jsonl`, summaries, config fingerprints)

### Out of Scope (Phase 1)

- reproducing LongMemEval retrieval pipeline internals exactly
- `longmemeval_m_cleaned.json` full sweep at first pass (cost/runtime heavy)

## Harness Architecture

Create a local benchmark package:

`tools/longmemeval/`

1) `prepare_data.py`
- downloads dataset splits into `benchmark_data/longmemeval/`
- records checksum and source URLs in metadata

2) `run_adk_longmemeval.py`
- loads ADK agent module path (`agent.py` package form)
- for each question instance:
  - creates a stable session id (e.g., `longmemeval-{question_id}`)
  - preloads the history sessions into session events in timestamp order
  - sends the benchmark question as a user turn via `Runner.run_async`
  - captures final response text
- writes `outputs/{run_id}/predictions.jsonl` with:
  - `question_id`
  - `hypothesis`
  - optional diagnostics (`latency_ms`, `session_id`)

3) `evaluate_qa.py`
- runs LongMemEval-style QA judging on predictions
- supports:
  - OpenAI judge model path (parity with LongMemEval script)
  - optional exact-match mode for smoke tests
- writes:
  - `outputs/{run_id}/qa_eval.jsonl`
  - `outputs/{run_id}/summary.json`

4) `compare_runs.py`
- compares multiple run summaries and prints deltas by question type

5) `README.md`
- one-command quickstart and full runbook

## Data and Session Mapping

LongMemEval fields used:
- `question_id`, `question_type`, `question`, `answer`
- `haystack_dates`, `haystack_sessions`, `answer_session_ids`

ADK mapping strategy:
- each haystack turn becomes a session event preserving role (`user`/`assistant`)
- timestamps follow `haystack_dates` ordering
- benchmark question is injected as the final user turn

Note:
- This preserves benchmark chronology while avoiding expensive model generation
  for history turns.

## Experiment Matrix

Run the same split across three configurations:

1) `baseline`
- no hybrid deterministic compaction

2) `hybrid`
- `enable_deterministic_compaction=True`
- `enable_observational_memory=False`

3) `hybrid_observational`
- `enable_deterministic_compaction=True`
- `enable_observational_memory=True`
- trigger metadata enabled for observational runtime in benchmark runs

Optional fourth configuration:
4) `hybrid_observational_no_trigger`
- validates no-op behavior when trigger metadata is absent

## Suggested CLI Commands

Data prep:

```bash
.venv/bin/python tools/longmemeval/prepare_data.py --out benchmark_data/longmemeval
```

Run one split:

```bash
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path path/to/agent_pkg \
  --dataset benchmark_data/longmemeval/longmemeval_s_cleaned.json \
  --config_name hybrid_observational \
  --out outputs/longmemeval/run_hybrid_obs_s
```

Evaluate:

```bash
OPENAI_API_KEY=... .venv/bin/python tools/longmemeval/evaluate_qa.py \
  --predictions outputs/longmemeval/run_hybrid_obs_s/predictions.jsonl \
  --reference benchmark_data/longmemeval/longmemeval_s_cleaned.json \
  --judge_model gpt-4o
```

Compare:

```bash
.venv/bin/python tools/longmemeval/compare_runs.py \
  --runs outputs/longmemeval/run_baseline_s \
         outputs/longmemeval/run_hybrid_s \
         outputs/longmemeval/run_hybrid_obs_s
```

## Reporting Format

For each run, publish:
- overall accuracy
- accuracy by `question_type`
- abstention accuracy (`question_id` suffix `_abs`)
- optional p50/p95 latency
- run metadata (commit hash, model, config, split)

Store report in:
- `tasks/longmemeval-results-<date>.md`

## Quality Gates for Adoption

Use this benchmark as a release gate only after:
- deterministic harness reproducibility verified on a fixed sample
- judge model fixed and versioned in report metadata
- at least one repeated run shows low variance on same config

## Delivery Plan

1) Implement harness scripts and README.
2) Smoke test on 20-example sample from `oracle` split.
3) Run full `oracle` split for all three configs.
4) Run `s_cleaned` split for all three configs.
5) Publish comparison report and recommendations.

## Risks and Mitigations

- LLM-as-judge variance
  - Mitigation: fixed judge model/version and optional repeated scoring.
- Runtime/cost on long histories
  - Mitigation: start with `oracle`, then `s_cleaned`; defer `m_cleaned`.
- Role/history ingestion mismatch
  - Mitigation: explicit event-role mapping tests in harness unit tests.
