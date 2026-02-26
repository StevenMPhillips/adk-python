# LongMemEval Harness Results (2026-02-26)

## Scope

This run validates the newly added harness plumbing and executes a reduced
benchmark matrix without external model APIs.

Artifacts and scripts used:
- `tools/longmemeval/prepare_data.py`
- `tools/longmemeval/run_adk_longmemeval.py`
- `tools/longmemeval/evaluate_qa.py`
- `tools/longmemeval/compare_runs.py`

## Environment and Setup

- Dataset downloaded to: `/tmp/longmemeval_data`
- Harness outputs written to: `/tmp/longmemeval_runs`
- Replay agent: `/tmp/longmemeval_smoke/hybrid_mock_agent.py`
  - deterministic callback response (`"stub hypothesis"`)
  - hybrid compaction config enabled for config toggling

## Commands Executed

Data preparation:

```bash
.venv/bin/python tools/longmemeval/prepare_data.py --out /tmp/longmemeval_data
```

Oracle matrix (25 cases each):

```bash
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py --agent_module_path /tmp/longmemeval_smoke/hybrid_mock_agent.py --dataset /tmp/longmemeval_data/longmemeval_oracle.json --out /tmp/longmemeval_runs/longmemeval_oracle_baseline_25 --config_name baseline --max_cases 25
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py --agent_module_path /tmp/longmemeval_smoke/hybrid_mock_agent.py --dataset /tmp/longmemeval_data/longmemeval_oracle.json --out /tmp/longmemeval_runs/longmemeval_oracle_hybrid_25 --config_name hybrid --max_cases 25
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py --agent_module_path /tmp/longmemeval_smoke/hybrid_mock_agent.py --dataset /tmp/longmemeval_data/longmemeval_oracle.json --out /tmp/longmemeval_runs/longmemeval_oracle_hybrid_observational_25 --config_name hybrid_observational --max_cases 25
```

S-cleaned matrix (3 cases each, reduced due runtime and local key constraints):

```bash
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py --agent_module_path /tmp/longmemeval_smoke/hybrid_mock_agent.py --dataset /tmp/longmemeval_data/longmemeval_s_cleaned.json --out /tmp/longmemeval_runs/longmemeval_s_cleaned_baseline_3 --config_name baseline --max_cases 3
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py --agent_module_path /tmp/longmemeval_smoke/hybrid_mock_agent.py --dataset /tmp/longmemeval_data/longmemeval_s_cleaned.json --out /tmp/longmemeval_runs/longmemeval_s_cleaned_hybrid_3 --config_name hybrid --max_cases 3
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py --agent_module_path /tmp/longmemeval_smoke/hybrid_mock_agent.py --dataset /tmp/longmemeval_data/longmemeval_s_cleaned.json --out /tmp/longmemeval_runs/longmemeval_s_cleaned_hybrid_observational_3 --config_name hybrid_observational --max_cases 3
```

Exact-mode evaluation for all generated runs:

```bash
.venv/bin/python tools/longmemeval/evaluate_qa.py --predictions <run>/predictions.jsonl --reference <split>.json --out_dir <run>/eval_exact --judge_mode exact
```

Run comparison outputs:

```bash
.venv/bin/python tools/longmemeval/compare_runs.py ... --json_out /tmp/longmemeval_runs/oracle_compare_exact.json
.venv/bin/python tools/longmemeval/compare_runs.py ... --json_out /tmp/longmemeval_runs/s_cleaned_compare_exact.json
```

## Results Summary

- Harness scripts completed end-to-end for both splits.
- Outputs present for:
  - oracle: baseline/hybrid/hybrid_observational (25 cases each)
  - s_cleaned: baseline/hybrid/hybrid_observational (3 cases each)
- Exact-mode metrics are all zeros for all runs, as expected for a mock agent
  returning fixed text.

## Observations and Limitations

1) **Harness functionality validated**
   - Data download, replay, prediction export, evaluation, and comparison all
     work.

2) **Observational runtime behavior observed**
   - `hybrid_observational` attempts observation generation and degrades
     gracefully when Gemini API keys are not set.

3) **Not a production-quality score run yet**
   - This is a plumbing/smoke benchmark run with a mock replay agent and
     exact-match evaluator.
   - It does not represent actual task quality for baseline vs hybrid configs.

## Recommendation

Proceed to a full-quality run with a real target agent and external evaluators:

1) Provide model credentials (`GOOGLE_API_KEY` for generation and
   `OPENAI_API_KEY` for LLM-as-judge), or configure equivalent local providers.
2) Re-run full matrix for oracle and s_cleaned with meaningful agent outputs.
3) Publish final recommendation after openai-judge evaluation and per-type
   deltas.
