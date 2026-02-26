# LongMemEval Harness Runbook

This directory contains a small CLI harness for running ADK agents on
LongMemEval-style data, grading QA outputs, and comparing run summaries.

## Quickstart

```bash
# 1) Download cleaned LongMemEval files.
.venv/bin/python tools/longmemeval/prepare_data.py

# 2) Run one inference configuration.
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path path/to/agent.py \
  --dataset tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out tools/longmemeval/runs/baseline_m \
  --config_name baseline

# 3) Evaluate predictions (OpenAI judge mode).
OPENAI_API_KEY=... .venv/bin/python tools/longmemeval/evaluate_qa.py \
  --predictions tools/longmemeval/runs/baseline_m/predictions.jsonl \
  --reference tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out_dir tools/longmemeval/runs/baseline_m/eval_openai \
  --judge_mode openai \
  --judge_model gpt-4o-mini

# 4) Compare multiple evaluated runs.
.venv/bin/python tools/longmemeval/compare_runs.py \
  tools/longmemeval/runs/baseline_m/eval_openai \
  tools/longmemeval/runs/hybrid_m/eval_openai \
  tools/longmemeval/runs/hybrid_observational_m/eval_openai
```

## Setup Prerequisites

- Python environment with ADK dependencies installed (recommended: project
  `.venv`).
- For `--judge_mode openai`, set `OPENAI_API_KEY` and install the `openai`
  package.
- An agent module that exports either:
  - `app` (an `App` instance), or
  - `root_agent` (an ADK agent instance)
- Use `.venv/bin/python` in commands to match repository dependencies.

## Data Preparation

Download all cleaned dataset files into `tools/longmemeval/data`:

```bash
.venv/bin/python tools/longmemeval/prepare_data.py
```

Optional flags:

```bash
.venv/bin/python tools/longmemeval/prepare_data.py \
  --out tools/longmemeval/data \
  --force
```

## Run Inference

Command template:

```bash
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path path/to/agent.py \
  --dataset tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out tools/longmemeval/runs/<run_name> \
  --config_name <baseline|hybrid|hybrid_observational> \
  [--max_cases 100]
```

Config matrix:

- `baseline`: deterministic compaction off, hybrid prompt assembly off,
  observational memory off.
- `hybrid`: deterministic compaction on, hybrid prompt assembly on,
  observational memory off.
- `hybrid_observational`: deterministic compaction on, hybrid prompt assembly
  on, observational memory on (plus runtime metadata trigger).

Example runs:

```bash
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path path/to/agent.py \
  --dataset tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out tools/longmemeval/runs/baseline_m \
  --config_name baseline

.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path path/to/agent.py \
  --dataset tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out tools/longmemeval/runs/hybrid_m \
  --config_name hybrid

.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path path/to/agent.py \
  --dataset tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out tools/longmemeval/runs/hybrid_observational_m \
  --config_name hybrid_observational
```

## QA Evaluation

Evaluate one prediction file against a reference split.

OpenAI mode:

```bash
OPENAI_API_KEY=... .venv/bin/python tools/longmemeval/evaluate_qa.py \
  --predictions tools/longmemeval/runs/baseline_m/predictions.jsonl \
  --reference tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out_dir tools/longmemeval/runs/baseline_m/eval_openai \
  --judge_mode openai \
  --judge_model gpt-4o-mini
```

Exact mode (deterministic fallback):

```bash
.venv/bin/python tools/longmemeval/evaluate_qa.py \
  --predictions tools/longmemeval/runs/baseline_m/predictions.jsonl \
  --reference tools/longmemeval/data/longmemeval_m_cleaned.json \
  --out_dir tools/longmemeval/runs/baseline_m/eval_exact \
  --judge_mode exact
```

## Compare Runs

Compare summaries side by side (first run is the baseline for deltas):

```bash
.venv/bin/python tools/longmemeval/compare_runs.py \
  tools/longmemeval/runs/baseline_m/eval_openai \
  tools/longmemeval/runs/hybrid_m/eval_openai \
  tools/longmemeval/runs/hybrid_observational_m/eval_openai
```

Optional machine-readable output:

```bash
.venv/bin/python tools/longmemeval/compare_runs.py \
  tools/longmemeval/runs/baseline_m/eval_openai \
  tools/longmemeval/runs/hybrid_m/eval_openai \
  --json_out tools/longmemeval/runs/compare_openai_m.json
```

## Output File Structure

Typical layout:

```text
tools/longmemeval/
  data/
    longmemeval_oracle.json
    longmemeval_s_cleaned.json
    longmemeval_m_cleaned.json
    metadata.json
  runs/
    baseline_m/
      predictions.jsonl
      run_metadata.json
      eval_openai/
        qa_eval.jsonl
        summary.json
      eval_exact/
        qa_eval.jsonl
        summary.json
```

Generated artifacts:

- `predictions.jsonl`: one JSON object per case with `question_id` and
  `hypothesis`.
- `run_metadata.json`: run configuration metadata (dataset path, config name,
  processed count, git commit hash, etc.).
- `qa_eval.jsonl`: per-example grading output with correctness labels and
  reasons.
- `summary.json`: aggregate accuracy, abstention metrics, and per-type metrics.

## Troubleshooting

- Missing key (`OPENAI_API_KEY`): `evaluate_qa.py --judge_mode openai` fails
  unless `OPENAI_API_KEY` is present in the environment.
- Dataset format errors:
  - `run_adk_longmemeval.py` accepts JSONL objects per line, a JSON list of
    objects, or top-level `data`/`items`/`examples` lists.
  - `evaluate_qa.py` accepts JSONL objects per line, a JSON list of objects, or
    top-level `data`/`items`/`predictions`/`references`/`examples` lists.
- No final response text (`hypothesis` is empty): the harness extracts the last
  non-user text event. Empty output usually means the agent never emitted final
  text parts for the query. Verify the agent produces a text response in the
  final turn.

## Reproducibility Checklist

- Pin and record the code revision (`git rev-parse HEAD`), and keep
  `run_metadata.json` with results.
- Record the exact QA judge setup: `judge_mode` and `judge_model`.
- Record dataset split and file path (`longmemeval_oracle`,
  `longmemeval_s_cleaned`, or `longmemeval_m_cleaned`).
