from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections import defaultdict
from typing import Any

try:
  from tools.longmemeval._common import get_first
  from tools.longmemeval._common import infer_should_abstain
  from tools.longmemeval._common import is_abstention_text
  from tools.longmemeval._common import load_json_or_jsonl
  from tools.longmemeval._common import normalize_text
  from tools.longmemeval._common import record_id
  from tools.longmemeval._common import safe_divide
  from tools.longmemeval._common import to_answer_list
  from tools.longmemeval._common import write_jsonl
except ModuleNotFoundError:
  from _common import get_first
  from _common import infer_should_abstain
  from _common import is_abstention_text
  from _common import load_json_or_jsonl
  from _common import normalize_text
  from _common import record_id
  from _common import safe_divide
  from _common import to_answer_list
  from _common import write_jsonl


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Evaluate LongMemEval-style QA predictions against references."
  )
  parser.add_argument(
      "--predictions", required=True, help="Path to model predictions (json/jsonl)."
  )
  parser.add_argument(
      "--reference", required=True, help="Path to references (json/jsonl)."
  )
  parser.add_argument(
      "--out_dir", required=True, help="Directory to write qa_eval.jsonl and summary.json."
  )
  parser.add_argument(
      "--judge_mode",
      choices=("openai", "exact"),
      default="openai",
      help="Judge mode: openai LLM judge or exact deterministic fallback.",
  )
  parser.add_argument(
      "--judge_model",
      default="gpt-4o-mini",
      help="OpenAI model for --judge_mode openai.",
  )
  return parser.parse_args()


def _question_type_instruction(question_type: str) -> str:
  normalized_type = normalize_text(question_type)
  rules_by_type = {
      "single hop": (
          "Check factual exactness for a single fact. Accept semantically equivalent"
          " paraphrases."
      ),
      "multi hop": (
          "Require all reasoning hops to be supported by the reference answer(s)."
          " Partial chains are incorrect."
      ),
      "temporal": (
          "Pay special attention to dates, time ordering, and recency. Incorrect"
          " temporal detail makes the answer incorrect."
      ),
      "yes no": (
          "For binary questions, match the polarity exactly. Unsupported hedges are"
          " incorrect."
      ),
      "entity": (
          "For entity extraction questions, entity identity must match; formatting"
          " differences are acceptable."
      ),
  }
  for key, rule in rules_by_type.items():
    if key in normalized_type:
      return rule
  return (
      "Judge semantic equivalence carefully. Reward faithful paraphrases and"
      " reject unsupported additions."
  )


def _exact_judge(
    *,
    question: str,
    question_type: str,
    reference_answers: list[str],
    prediction: str,
    should_abstain: bool,
) -> dict[str, Any]:
  del question
  del question_type
  predicted_abstain = is_abstention_text(prediction)

  if should_abstain:
    is_correct = predicted_abstain
    return {
        "label": "CORRECT" if is_correct else "INCORRECT",
        "is_correct": is_correct,
        "predicted_abstain": predicted_abstain,
        "reason": "Expected abstention." if is_correct else "Abstention expected.",
    }

  normalized_prediction = normalize_text(prediction)
  normalized_refs = {
      normalize_text(answer) for answer in reference_answers if answer.strip()
  }
  is_correct = (
      bool(normalized_prediction)
      and not predicted_abstain
      and normalized_prediction in normalized_refs
  )
  return {
      "label": "CORRECT" if is_correct else "INCORRECT",
      "is_correct": is_correct,
      "predicted_abstain": predicted_abstain,
      "reason": "Exact normalized match." if is_correct else "No exact normalized match.",
  }


def _openai_judge(
    *,
    question: str,
    question_type: str,
    reference_answers: list[str],
    prediction: str,
    should_abstain: bool,
    judge_model: str,
) -> dict[str, Any]:
  try:
    from openai import OpenAI
  except ImportError as exc:
    raise RuntimeError(
        "openai package is required for --judge_mode openai."
    ) from exc

  if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is required for --judge_mode openai.")

  system_prompt = (
      "You are a strict LongMemEval QA grader. Return only valid JSON with keys"
      " label, predicted_abstain, and reason. label must be CORRECT or INCORRECT."
      " predicted_abstain must be true/false."
  )

  instruction = _question_type_instruction(question_type)
  user_prompt = {
      "grading_rule": instruction,
      "question_type": question_type,
      "question": question,
      "reference_answers": reference_answers,
      "should_abstain": should_abstain,
      "model_prediction": prediction,
      "additional_constraints": (
          "If should_abstain is true, CORRECT only when the prediction abstains."
          " If should_abstain is false, abstentions are INCORRECT."
      ),
      "output_schema": {
          "label": "CORRECT|INCORRECT",
          "predicted_abstain": "boolean",
          "reason": "short explanation",
      },
  }

  client = OpenAI()
  response = client.chat.completions.create(
      model=judge_model,
      temperature=0,
      response_format={"type": "json_object"},
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
      ],
  )
  message = response.choices[0].message.content or "{}"
  parsed = json.loads(message)
  label = str(parsed.get("label", "INCORRECT")).upper().strip()
  if label not in ("CORRECT", "INCORRECT"):
    label = "INCORRECT"
  predicted_abstain = bool(parsed.get("predicted_abstain", is_abstention_text(prediction)))
  is_correct = label == "CORRECT"
  return {
      "label": label,
      "is_correct": is_correct,
      "predicted_abstain": predicted_abstain,
      "reason": str(parsed.get("reason", "")),
  }


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  total = len(rows)
  correct = sum(1 for row in rows if row["is_correct"])

  abstention_candidates = [row for row in rows if row["should_abstain"]]
  abstention_correct = sum(1 for row in abstention_candidates if row["is_correct"])

  type_counts: dict[str, int] = defaultdict(int)
  type_correct: dict[str, int] = defaultdict(int)
  for row in rows:
    question_type = row["question_type"]
    type_counts[question_type] += 1
    if row["is_correct"]:
      type_correct[question_type] += 1

  per_type: dict[str, dict[str, Any]] = {}
  for question_type in sorted(type_counts):
    count = type_counts[question_type]
    num_correct = type_correct.get(question_type, 0)
    per_type[question_type] = {
        "count": count,
        "correct": num_correct,
        "accuracy": safe_divide(num_correct, count),
    }

  return {
      "num_examples": total,
      "overall_accuracy": safe_divide(correct, total),
      "abstention_accuracy": safe_divide(
          abstention_correct, len(abstention_candidates)
      ),
      "abstention_count": len(abstention_candidates),
      "per_question_type_accuracy": per_type,
  }


def main() -> None:
  args = _parse_args()
  predictions_path = pathlib.Path(args.predictions)
  reference_path = pathlib.Path(args.reference)
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  predictions = load_json_or_jsonl(predictions_path)
  references = load_json_or_jsonl(reference_path)

  prediction_by_id: dict[str, dict[str, Any]] = {}
  for index, record in enumerate(predictions):
    prediction_by_id[record_id(record, index)] = record

  rows: list[dict[str, Any]] = []
  for index, reference in enumerate(references):
    item_id = record_id(reference, index)
    prediction_record = prediction_by_id.get(item_id)
    if prediction_record is None:
      prediction_text = ""
    else:
      prediction_text = str(
          get_first(
              prediction_record,
              ("prediction", "pred", "answer", "response", "output", "text"),
              default="",
          )
      )

    question = str(get_first(reference, ("question", "query", "prompt"), ""))
    question_type = str(
        get_first(reference, ("question_type", "type", "category"), "unknown")
    )
    reference_answers = to_answer_list(
        get_first(
            reference,
            (
                "answers",
                "answer",
                "ground_truth",
                "reference_answers",
                "reference",
                "gold",
            ),
        )
    )
    should_abstain = infer_should_abstain(reference, reference_answers)

    if args.judge_mode == "exact":
      judge_result = _exact_judge(
          question=question,
          question_type=question_type,
          reference_answers=reference_answers,
          prediction=prediction_text,
          should_abstain=should_abstain,
      )
    else:
      judge_result = _openai_judge(
          question=question,
          question_type=question_type,
          reference_answers=reference_answers,
          prediction=prediction_text,
          should_abstain=should_abstain,
          judge_model=args.judge_model,
      )

    rows.append(
      {
          "id": item_id,
          "question": question,
          "question_type": question_type,
          "reference_answers": reference_answers,
          "prediction": prediction_text,
          "should_abstain": should_abstain,
          "predicted_abstain": judge_result["predicted_abstain"],
          "label": judge_result["label"],
          "is_correct": judge_result["is_correct"],
          "reason": judge_result["reason"],
      }
    )

  qa_eval_path = out_dir / "qa_eval.jsonl"
  summary_path = out_dir / "summary.json"
  write_jsonl(qa_eval_path, rows)
  summary = _build_summary(rows)
  summary.update({
      "judge_mode": args.judge_mode,
      "judge_model": args.judge_model if args.judge_mode == "openai" else None,
      "predictions_path": str(predictions_path),
      "reference_path": str(reference_path),
  })
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

  print(f"Wrote {qa_eval_path}")
  print(f"Wrote {summary_path}")


if __name__ == "__main__":
  main()
