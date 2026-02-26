from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compare LongMemEval run summaries side by side."
  )
  parser.add_argument(
      "run_dirs",
      nargs="+",
      help="One or more run directories that contain summary.json.",
  )
  parser.add_argument(
      "--json_out",
      default=None,
      help="Optional path for machine-readable comparison JSON output.",
  )
  return parser.parse_args()


def _load_summary(run_dir: pathlib.Path) -> dict[str, Any]:
  summary_path = run_dir / "summary.json"
  if not summary_path.exists():
    raise FileNotFoundError(f"Missing summary.json in {run_dir}")
  return json.loads(summary_path.read_text(encoding="utf-8"))


def _fmt_float(value: float | None) -> str:
  if value is None:
    return "n/a"
  return f"{value:.4f}"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
  widths = [len(header) for header in headers]
  for row in rows:
    for i, cell in enumerate(row):
      widths[i] = max(widths[i], len(cell))

  def render(row: list[str]) -> str:
    return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

  print(render(headers))
  print("-+-".join("-" * width for width in widths))
  for row in rows:
    print(render(row))


def _delta(current: float | None, base: float | None) -> str:
  if current is None or base is None:
    return "n/a"
  return f"{(current - base):+.4f}"


def main() -> None:
  args = _parse_args()
  run_dirs = [pathlib.Path(path) for path in args.run_dirs]

  summaries: list[dict[str, Any]] = []
  for run_dir in run_dirs:
    summaries.append(_load_summary(run_dir))

  base = summaries[0]
  base_overall = base.get("overall_accuracy")
  base_abstain = base.get("abstention_accuracy")

  metric_rows: list[list[str]] = []
  for run_dir, summary in zip(run_dirs, summaries):
    overall = summary.get("overall_accuracy")
    abstain = summary.get("abstention_accuracy")
    metric_rows.append(
        [
            str(run_dir),
            _fmt_float(overall),
            _fmt_float(abstain),
            _delta(overall, base_overall),
            _delta(abstain, base_abstain),
        ]
    )

  print("== Overall Metrics ==")
  _print_table(
      headers=[
          "run_dir",
          "overall_accuracy",
          "abstention_accuracy",
          "delta_overall_vs_base",
          "delta_abstention_vs_base",
      ],
      rows=metric_rows,
  )

  all_question_types: set[str] = set()
  for summary in summaries:
    all_question_types.update(summary.get("per_question_type_accuracy", {}).keys())

  if all_question_types:
    print("\n== Per Question Type Accuracy ==")
    for question_type in sorted(all_question_types):
      rows: list[list[str]] = []
      base_metric = (
          base.get("per_question_type_accuracy", {})
          .get(question_type, {})
          .get("accuracy")
      )
      for run_dir, summary in zip(run_dirs, summaries):
        metric = (
            summary.get("per_question_type_accuracy", {})
            .get(question_type, {})
            .get("accuracy")
        )
        rows.append(
            [
                str(run_dir),
                _fmt_float(metric),
                _delta(metric, base_metric),
            ]
        )
      print(f"\n[{question_type}]")
      _print_table(
          headers=["run_dir", "accuracy", "delta_vs_base"],
          rows=rows,
      )

  if args.json_out:
    output = {
        "base_run": str(run_dirs[0]),
        "runs": [
            {
                "run_dir": str(run_dir),
                "overall_accuracy": summary.get("overall_accuracy"),
                "abstention_accuracy": summary.get("abstention_accuracy"),
                "per_question_type_accuracy": summary.get("per_question_type_accuracy", {}),
                "delta_overall_vs_base": (
                    None
                    if summary.get("overall_accuracy") is None or base_overall is None
                    else summary.get("overall_accuracy") - base_overall
                ),
                "delta_abstention_vs_base": (
                    None
                    if summary.get("abstention_accuracy") is None or base_abstain is None
                    else summary.get("abstention_accuracy") - base_abstain
                ),
            }
            for run_dir, summary in zip(run_dirs, summaries)
        ],
    }
    out_path = pathlib.Path(args.json_out)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
  main()
