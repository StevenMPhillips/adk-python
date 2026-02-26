from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import pathlib
import subprocess
import sys
import uuid
from typing import Any

from google.genai import types

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.compaction.config import HybridEventsCompactionConfig
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.longmemeval._replay_utils import extract_text
from tools.longmemeval._replay_utils import load_module_from_path
from tools.longmemeval._replay_utils import normalize_role
from tools.longmemeval._replay_utils import parse_timestamp


_VALID_CONFIGS = ('baseline', 'hybrid', 'hybrid_observational')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Run ADK agent replay on LongMemEval split.'
  )
  parser.add_argument(
      '--agent_module_path',
      required=True,
      help='Import path or file path for module exporting app/root_agent.',
  )
  parser.add_argument(
      '--dataset',
      type=pathlib.Path,
      required=True,
      help='Path to LongMemEval split JSON/JSONL file.',
  )
  parser.add_argument(
      '--out',
      type=pathlib.Path,
      required=True,
      help='Output directory for predictions and run metadata.',
  )
  parser.add_argument(
      '--config_name',
      choices=_VALID_CONFIGS,
      required=True,
      help='Runtime config mode.',
  )
  parser.add_argument(
      '--max_cases',
      type=int,
      default=None,
      help='Optional cap on number of cases to run.',
  )
  return parser.parse_args()


def _load_cases(path: pathlib.Path) -> list[dict[str, Any]]:
  text = path.read_text(encoding='utf-8')
  if path.suffix == '.jsonl':
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      if not isinstance(item, dict):
        raise ValueError(f'Expected json object per line in {path}.')
      rows.append(item)
    return rows

  parsed = json.loads(text)
  if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
    return parsed
  if isinstance(parsed, dict):
    for key in ('data', 'items', 'examples'):
      value = parsed.get(key)
      if isinstance(value, list) and all(
          isinstance(item, dict) for item in value
      ):
        return value
  raise ValueError(f'Unsupported dataset format in {path}.')


def _load_app_or_agent(agent_module_path: str) -> App:
  module = load_module_from_path(agent_module_path)
  app = getattr(module, 'app', None)
  if app is not None:
    if not isinstance(app, App):
      raise ValueError('Module "app" must be an instance of App.')
    return app.model_copy(deep=True)

  root_agent = getattr(module, 'root_agent', None)
  if root_agent is None:
    raise ValueError(
        'Module must define either "app" or "root_agent" variable.'
    )
  if not isinstance(root_agent, BaseAgent):
    raise ValueError('Module "root_agent" must be an ADK agent instance.')
  app_name = f'{root_agent.name}_app'
  return App(name=app_name, root_agent=root_agent)


def _configure_app(app: App, config_name: str) -> App:
  configured = app.model_copy(deep=True)
  config = configured.events_compaction_config
  if not isinstance(config, HybridEventsCompactionConfig):
    return configured

  if config_name == 'baseline':
    config.enable_deterministic_compaction = False
    config.enable_hybrid_prompt_assembly = False
    config.enable_observational_memory = False
  elif config_name == 'hybrid':
    config.enable_deterministic_compaction = True
    config.enable_hybrid_prompt_assembly = True
    config.enable_observational_memory = False
  elif config_name == 'hybrid_observational':
    config.enable_deterministic_compaction = True
    config.enable_hybrid_prompt_assembly = True
    config.enable_observational_memory = True
  return configured


def _collect_history_messages(
    case: dict[str, Any],
) -> list[tuple[float, int, str, str]]:
  history: list[tuple[float, int, str, str]] = []
  haystack_sessions = case.get('haystack_sessions')

  if isinstance(haystack_sessions, list):
    haystack_dates = case.get('haystack_dates', [])
    for session_index, session in enumerate(haystack_sessions):
      session_ts = None
      if isinstance(haystack_dates, list) and session_index < len(haystack_dates):
        session_ts = parse_timestamp(haystack_dates[session_index])
      if session_ts is None:
        session_ts = float(session_index)

      if not isinstance(session, list):
        continue
      for message_index, item in enumerate(session):
        if not isinstance(item, dict):
          continue
        text = extract_text(item)
        if not text:
          continue
        role = normalize_role(item.get('role') or item.get('speaker'))
        ts = parse_timestamp(item.get('timestamp') or item.get('date'))
        if ts is None:
          ts = session_ts + (message_index * 0.001)
        history.append((ts, message_index, role, text))
    history.sort(key=lambda item: (item[0], item[1]))
    return history

  return history


def _extract_question(case: dict[str, Any]) -> str:
  for key in ('question', 'query', 'prompt'):
    value = case.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return ''


def _extract_question_id(case: dict[str, Any], index: int) -> str:
  for key in ('question_id', 'id', 'qid', 'example_id'):
    value = case.get(key)
    if value is not None:
      return str(value)
  return f'case_{index}'


def _text_from_event(event: Event) -> str:
  if event.content is None or not event.content.parts:
    return ''
  return ''.join(part.text or '' for part in event.content.parts).strip()


def _extract_hypothesis(events: list[Event]) -> str:
  for event in reversed(events):
    if event.author == 'user':
      continue
    text = _text_from_event(event)
    if text:
      return text
  return ''


def _build_run_config(config_name: str) -> RunConfig | None:
  if config_name != 'hybrid_observational':
    return None
  return RunConfig(
      custom_metadata={'trigger_observational_memory_runtime': True}
  )


def _get_commit_hash(repo_root: pathlib.Path) -> str | None:
  try:
    output = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_root,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return output.strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return None


async def _run_case(
    *,
    runner: InMemoryRunner,
    app: App,
    case: dict[str, Any],
    index: int,
    run_config: RunConfig | None,
) -> dict[str, Any]:
  question_id = _extract_question_id(case, index)
  question = _extract_question(case)

  session = await runner.session_service.create_session(
      app_name=app.name,
      user_id='longmemeval_user',
      session_id=f'{question_id}_{uuid.uuid4().hex[:8]}',
  )

  history = _collect_history_messages(case)
  assistant_author = app.root_agent.name
  for event_index, (timestamp, _, role, text) in enumerate(history):
    author = 'user' if role == 'user' else assistant_author
    content_role = 'user' if role == 'user' else 'model'
    replay_event = Event(
        invocation_id=f'replay_{question_id}_{event_index}',
        author=author,
        content=types.Content(role=content_role, parts=[types.Part(text=text)]),
        timestamp=timestamp,
    )
    await runner.session_service.append_event(session=session, event=replay_event)

  events: list[Event] = []
  async for event in runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=types.UserContent(parts=[types.Part(text=question)]),
      run_config=run_config,
  ):
    events.append(event)

  return {
      'question_id': question_id,
      'hypothesis': _extract_hypothesis(events),
  }


async def main_async() -> None:
  args = parse_args()
  out_dir = args.out.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  app = _configure_app(_load_app_or_agent(args.agent_module_path), args.config_name)
  cases = _load_cases(args.dataset.resolve())
  if args.max_cases is not None:
    cases = cases[: args.max_cases]

  run_config = _build_run_config(args.config_name)
  predictions_path = out_dir / 'predictions.jsonl'
  repo_root = pathlib.Path(__file__).resolve().parents[2]

  processed_count = 0
  with predictions_path.open('w', encoding='utf-8') as out_file:
    async with InMemoryRunner(app=app) as runner:
      for index, case in enumerate(cases, start=1):
        prediction = await _run_case(
            runner=runner,
            app=app,
            case=case,
            index=index,
            run_config=run_config,
        )
        out_file.write(json.dumps(prediction, ensure_ascii=True, sort_keys=True))
        out_file.write('\n')
        processed_count += 1

  metadata = {
      'dataset_path': str(args.dataset.resolve()),
      'config_name': args.config_name,
      'total_cases_loaded': len(cases),
      'processed_cases': processed_count,
      'predictions_path': str(predictions_path),
      'run_started_at_utc': datetime.datetime.now(
          datetime.timezone.utc
      ).isoformat(),
      'agent_module_path': args.agent_module_path,
      'observational_runtime_triggered': (
          args.config_name == 'hybrid_observational'
      ),
      'git_commit_hash': _get_commit_hash(repo_root),
  }
  metadata_path = out_dir / 'run_metadata.json'
  metadata_path.write_text(
      json.dumps(metadata, indent=2, sort_keys=True),
      encoding='utf-8',
  )
  print(f'Wrote predictions: {predictions_path}')
  print(f'Wrote metadata: {metadata_path}')


def main() -> None:
  asyncio.run(main_async())


if __name__ == '__main__':
  main()
