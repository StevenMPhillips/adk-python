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

## Final Credentialed Matrix (Completed)

All six full runs completed with the dedicated in-repo benchmark agent:

- `full_longmemeval_oracle_baseline` (500)
- `full_longmemeval_oracle_hybrid` (500)
- `full_longmemeval_oracle_hybrid_observational` (500)
- `full_longmemeval_s_cleaned_baseline` (500)
- `full_longmemeval_s_cleaned_hybrid` (500)
- `full_longmemeval_s_cleaned_hybrid_observational` (500)

OpenAI-judge summaries (`gpt-4o-mini`) were generated for all runs.

### Aggregate Results

Oracle split:
- baseline overall accuracy: **0.002**
- hybrid overall accuracy: **0.004**
- hybrid_observational overall accuracy: **0.002**

S-cleaned split:
- baseline overall accuracy: **0.000**
- hybrid overall accuracy: **0.002**
- hybrid_observational overall accuracy: **0.000**

Observed pattern:
- extremely low absolute accuracy across all modes with this dedicated agent.
- a small lift appears for `hybrid` vs `baseline` in `single-session-user`.
- no measurable lift from `hybrid_observational` with current agent/prompting.

### Interpretation

These numbers should be interpreted as **harness readiness**, not product
quality readiness:

- The harness and scoring pipeline are now operational end to end.
- The dedicated benchmark agent is intentionally simple and underpowered for
  LongMemEval difficulty, so absolute scores are not deployment-grade signals.

### Additional Notes

- During long `hybrid_observational` runs, some observation-writer generations
  still fail with invalid JSON (truncated model output). The runner degrades
  gracefully and continues, but this lowers observational artifact yield.
- JSONL output integrity needed one repair pass in one run due concatenated
  entries from interrupted resumptions; deduplication was applied by
  `question_id` before final scoring.

### Final Recommendation

Use this benchmark setup as the official ADK LongMemEval harness baseline, but
do not use the current dedicated sample agent scores as rollout criteria.

Next required step for decision-grade evaluation:
1) benchmark a realistic target agent (or tuned benchmark agent), and
2) tighten observational-writer structured-output robustness before relying on
   `hybrid_observational` comparisons.
