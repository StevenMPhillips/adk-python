from __future__ import annotations

import datetime
import importlib
import importlib.util
import pathlib
from typing import Any


_TIMESTAMP_FORMATS = (
    '%Y/%m/%d (%a) %H:%M',
    '%Y/%m/%d %H:%M',
)


def parse_timestamp(raw_value: Any) -> float | None:
  """Parses common timestamp values into Unix seconds."""
  if raw_value is None:
    return None
  if isinstance(raw_value, (int, float)):
    return float(raw_value)
  if not isinstance(raw_value, str):
    return None

  value = raw_value.strip()
  if not value:
    return None

  for fmt in _TIMESTAMP_FORMATS:
    try:
      return datetime.datetime.strptime(value, fmt).timestamp()
    except ValueError:
      continue

  normalized = value.replace('Z', '+00:00')
  try:
    return datetime.datetime.fromisoformat(normalized).timestamp()
  except ValueError:
    return None


def extract_text(value: Any) -> str:
  """Extracts readable text from common message shapes."""
  if value is None:
    return ''
  if isinstance(value, str):
    return value.strip()
  if isinstance(value, (int, float, bool)):
    return str(value)
  if isinstance(value, dict):
    for key in ('content', 'text', 'message', 'utterance', 'value'):
      if key in value:
        nested = extract_text(value[key])
        if nested:
          return nested
    return ''
  if isinstance(value, list):
    parts = [extract_text(item) for item in value]
    return '\n'.join(part for part in parts if part)
  return ''


def normalize_role(raw_role: Any) -> str:
  """Normalizes role labels to user/assistant/system."""
  if not isinstance(raw_role, str):
    return 'user'
  role = raw_role.strip().lower()
  if role in ('user', 'human'):
    return 'user'
  if role in ('assistant', 'model', 'bot', 'agent'):
    return 'assistant'
  if role in ('system',):
    return 'system'
  return 'user'


def load_module_from_path(module_path: str):
  """Loads an importable module path or Python file path."""
  path = pathlib.Path(module_path)
  if path.exists():
    module_name = f'longmemeval_agent_{path.stem}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
      raise ValueError(f'Unable to import module from path: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
  return importlib.import_module(module_path)
