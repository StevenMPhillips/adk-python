# LongMemEval Benchmark Agent

This sample agent is intended for LongMemEval harness runs.

It exports both:
- `root_agent`
- `app` with `HybridEventsCompactionConfig`

The harness toggles runtime modes via `--config_name`:
- `baseline`
- `hybrid`
- `hybrid_observational`

Example harness command:

```bash
.venv/bin/python tools/longmemeval/run_adk_longmemeval.py \
  --agent_module_path contributing/samples/longmemeval_benchmark_agent/agent.py \
  --dataset /tmp/longmemeval_data/longmemeval_oracle.json \
  --out /tmp/longmemeval_runs/lme_oracle_baseline \
  --config_name baseline \
  --max_cases 10
```
