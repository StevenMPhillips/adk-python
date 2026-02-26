from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterable
from typing import Any


_ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
    "not enough information",
    "insufficient information",
    "unknown",
    "unable to answer",
    "cannot answer",
    "can't answer",
)


def load_json_or_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
  """Loads records from a json or jsonl file."""
  text = path.read_text(encoding="utf-8")
  if path.suffix == ".jsonl":
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      if not isinstance(item, dict):
        raise ValueError(f"Expected JSON object per line in {path}.")
      records.append(item)
    return records

  parsed = json.loads(text)
  if isinstance(parsed, list):
    if not all(isinstance(item, dict) for item in parsed):
      raise ValueError(f"Expected a list of objects in {path}.")
    return parsed
  if isinstance(parsed, dict):
    for key in ("data", "items", "predictions", "references", "examples"):
      value = parsed.get(key)
      if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
  raise ValueError(f"Unable to parse records from {path}.")


def record_id(record: dict[str, Any], fallback_index: int) -> str:
  """Returns a stable record id string."""
  for key in ("id", "question_id", "qid", "example_id", "uuid"):
    value = record.get(key)
    if value is not None:
      return str(value)
  return str(fallback_index)


def get_first(record: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
  for key in keys:
    if key in record and record[key] is not None:
      return record[key]
  return default


def normalize_text(text: str) -> str:
  """Normalizes text for deterministic exact matching."""
  lowered = text.lower().strip()
  lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
  lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
  lowered = re.sub(r"\s+", " ", lowered)
  return lowered.strip()


def to_answer_list(raw_value: Any) -> list[str]:
  """Normalizes answer-like structures into a list of strings."""
  if raw_value is None:
    return []
  if isinstance(raw_value, str):
    return [raw_value]
  if isinstance(raw_value, (int, float, bool)):
    return [str(raw_value)]
  if isinstance(raw_value, dict):
    for key in ("answer", "answers", "text", "value", "gold"):
      if key in raw_value:
        return to_answer_list(raw_value[key])
    return [json.dumps(raw_value, sort_keys=True)]
  if isinstance(raw_value, list):
    values: list[str] = []
    for item in raw_value:
      values.extend(to_answer_list(item))
    return [value for value in values if value != ""]
  return [str(raw_value)]


def is_abstention_text(text: str) -> bool:
  """Returns True when text looks like an abstention response."""
  normalized = normalize_text(text)
  if not normalized:
    return True
  return any(pattern in normalized for pattern in _ABSTAIN_PATTERNS)


def infer_should_abstain(record: dict[str, Any], answers: list[str]) -> bool:
  """Infers if a question expects abstention."""
  maybe_bool = get_first(
      record,
      (
          "should_abstain",
          "requires_abstention",
          "is_unanswerable",
          "unanswerable",
          "answerable",
      ),
  )
  if isinstance(maybe_bool, bool):
    if "answerable" in record and maybe_bool is True:
      return False
    if "answerable" in record and maybe_bool is False:
      return True
    return maybe_bool
  return any(is_abstention_text(answer) for answer in answers)


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
  """Writes jsonl rows to disk."""
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
      f.write("\n")


def safe_divide(numerator: int, denominator: int) -> float:
  if denominator == 0:
    return 0.0
  return numerator / denominator
