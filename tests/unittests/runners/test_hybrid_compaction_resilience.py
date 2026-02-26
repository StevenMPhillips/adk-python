# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.compaction.compactors.patch_compactor import PatchCompactor
from google.adk.compaction.compactors.registry import ToolRunCompactorRegistry
from google.adk.compaction.config import HybridEventsCompactionConfig
from google.adk.compaction.storage.in_memory_compaction_service import InMemoryCompactionService
from google.adk.tools.function_tool import FunctionTool
from google.genai.types import FunctionCall
from google.genai.types import Part
import pytest

from .. import testing_utils


def _run_shell(command: str) -> dict[str, object]:
  return {
      'command': command,
      'exit_code': 0,
      'stdout': 'ok',
      'stderr': '',
      'duration_ms': 5,
  }


def _event_texts(events) -> list[str]:
  texts: list[str] = []
  for event in events:
    if not event.content or not event.content.parts:
      continue
    texts.extend(part.text for part in event.content.parts if part.text)
  return texts


class _FailingCompactionService(InMemoryCompactionService):

  def __init__(
      self,
      *,
      fail_save_tool_run: bool = False,
      fail_get_task_state: bool = False,
  ):
    super().__init__()
    self.fail_save_tool_run = fail_save_tool_run
    self.fail_get_task_state = fail_get_task_state
    self.save_tool_run_calls = 0
    self.get_task_state_calls = 0

  async def save_tool_run_compaction(self, tool_run_compaction) -> None:
    self.save_tool_run_calls += 1
    if self.fail_save_tool_run:
      raise RuntimeError('save_tool_run_compaction failed')
    await super().save_tool_run_compaction(tool_run_compaction)

  async def get_task_state(self, session_id: str):
    self.get_task_state_calls += 1
    if self.fail_get_task_state:
      raise RuntimeError('get_task_state failed')
    return await super().get_task_state(session_id)


class _NoOpCheckCompactionService(InMemoryCompactionService):

  def __init__(self):
    super().__init__()
    self.save_observation_calls = 0
    self.save_reflection_calls = 0
    self.save_task_state_calls = 0

  async def save_observation(self, observation) -> None:
    self.save_observation_calls += 1
    await super().save_observation(observation)

  async def save_reflection(self, reflection) -> None:
    self.save_reflection_calls += 1
    await super().save_reflection(reflection)

  async def save_task_state(self, task_state) -> None:
    self.save_task_state_calls += 1
    await super().save_task_state(task_state)


@pytest.mark.asyncio
async def test_hybrid_deterministic_save_failure_degrades_but_continues():
  tool = FunctionTool(func=_run_shell)
  model = testing_utils.MockModel.create([
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name,
                          args={'command': 'pytest tests/unittests'},
                      )
                  )
              ]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[Part(text='tool invocation completed')]
          )
      ),
  ])
  compaction_service = _FailingCompactionService(fail_save_tool_run=True)
  app = App(
      name='hybrid_resilience_app',
      root_agent=LlmAgent(name='agent', model=model, tools=[tool]),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async('run the shell tool')

  assert compaction_service.save_tool_run_calls >= 1
  assert 'tool invocation completed' in _event_texts(events)


@pytest.mark.asyncio
async def test_hybrid_deterministic_save_failure_emits_degradation_signal(
    caplog,
):
  tool = FunctionTool(func=_run_shell)
  model = testing_utils.MockModel.create([
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name,
                          args={'command': 'pytest tests/unittests'},
                      )
                  )
              ]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(parts=[Part(text='done')])
      ),
  ])
  compaction_service = _FailingCompactionService(fail_save_tool_run=True)
  app = App(
      name='hybrid_resilience_logging_app',
      root_agent=LlmAgent(name='agent', model=model, tools=[tool]),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  with caplog.at_level(logging.ERROR, logger='google_adk'):
    events = await runner.run_async('run the shell tool')

  assert 'done' in _event_texts(events)
  degradation_records = [
      record
      for record in caplog.records
      if (
          record.levelno == logging.ERROR
          and 'Deterministic tool-run compaction failed for session_id='
          in record.getMessage()
      )
  ]
  assert degradation_records
  assert all(
      'invocation_id=' in record.getMessage() for record in degradation_records
  )
  assert all(
      'event_id=' in record.getMessage() for record in degradation_records
  )


@pytest.mark.asyncio
async def test_hybrid_prompt_assembly_query_failure_degrades_but_continues():
  model = testing_utils.MockModel.create(['hybrid fallback response'])
  compaction_service = _FailingCompactionService(fail_get_task_state=True)
  app = App(
      name='hybrid_prompt_resilience_app',
      root_agent=LlmAgent(name='agent', model=model),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
          enable_hybrid_prompt_assembly=True,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async('hello hybrid prompt assembly')

  assert compaction_service.get_task_state_calls >= 1
  assert 'hybrid fallback response' in _event_texts(events)


@pytest.mark.asyncio
async def test_hybrid_prompt_assembly_query_failure_emits_degradation_signal(
    caplog,
):
  model = testing_utils.MockModel.create(['hybrid fallback response'])
  compaction_service = _FailingCompactionService(fail_get_task_state=True)
  app = App(
      name='hybrid_prompt_resilience_logging_app',
      root_agent=LlmAgent(name='agent', model=model),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
          enable_hybrid_prompt_assembly=True,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  with caplog.at_level(logging.ERROR, logger='google_adk'):
    events = await runner.run_async('hello hybrid prompt assembly')

  assert 'hybrid fallback response' in _event_texts(events)
  degradation_records = [
      record
      for record in caplog.records
      if (
          record.levelno == logging.ERROR
          and 'Hybrid prompt assembly failed for session_id='
          in record.getMessage()
      )
  ]
  assert degradation_records
  assert all(
      'invocation_id=' in record.getMessage() for record in degradation_records
  )


@pytest.mark.asyncio
async def test_non_hybrid_compaction_failure_remains_fail_fast(monkeypatch):
  model = testing_utils.MockModel.create(['response before compaction'])
  app = App(
      name='non_hybrid_fail_fast_app',
      root_agent=LlmAgent(name='agent', model=model),
      events_compaction_config=EventsCompactionConfig(
          compaction_interval=1,
          overlap_size=0,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  async def _raise_compaction_error(*_args, **_kwargs):
    raise RuntimeError('non-hybrid compaction failed')

  monkeypatch.setattr(
      'google.adk.runners._run_compaction_for_sliding_window',
      _raise_compaction_error,
  )

  with pytest.raises(RuntimeError, match='non-hybrid compaction failed'):
    await runner.run_async('trigger non-hybrid compaction')


@pytest.mark.asyncio
async def test_observational_runtime_not_initialized_when_flag_disabled(
    monkeypatch,
):
  model = testing_utils.MockModel.create(['response'])
  compaction_service = _NoOpCheckCompactionService()
  app = App(
      name='observational_runtime_disabled_app',
      root_agent=LlmAgent(name='agent', model=model),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
          enable_observational_memory=False,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  construction_calls = {'observation': 0, 'reflection': 0, 'task_state': 0}

  class _ObservationWriterStub:

    def __init__(self, **_kwargs):
      construction_calls['observation'] += 1

  class _ReflectionWriterStub:

    def __init__(self, **_kwargs):
      construction_calls['reflection'] += 1

  class _TaskStateUpdaterStub:

    def __init__(self, **_kwargs):
      construction_calls['task_state'] += 1

  monkeypatch.setattr(
      'google.adk.runners.ObservationWriter', _ObservationWriterStub
  )
  monkeypatch.setattr(
      'google.adk.runners.ReflectionWriter', _ReflectionWriterStub
  )
  monkeypatch.setattr(
      'google.adk.runners.TaskStateUpdater', _TaskStateUpdaterStub
  )

  events = await runner.run_async('run without observational runtime')

  assert 'response' in _event_texts(events)
  assert construction_calls == {
      'observation': 0,
      'reflection': 0,
      'task_state': 0,
  }


@pytest.mark.asyncio
async def test_observational_runtime_initializes_when_flag_enabled_and_is_noop(
    monkeypatch,
):
  model = testing_utils.MockModel.create(['response one', 'response two'])
  compaction_service = _NoOpCheckCompactionService()
  app = App(
      name='observational_runtime_enabled_app',
      root_agent=LlmAgent(name='agent', model=model),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
          enable_observational_memory=True,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  construction_calls = {'observation': 0, 'reflection': 0, 'task_state': 0}

  class _ObservationWriterStub:

    def __init__(self, **_kwargs):
      construction_calls['observation'] += 1

  class _ReflectionWriterStub:

    def __init__(self, **_kwargs):
      construction_calls['reflection'] += 1

  class _TaskStateUpdaterStub:

    def __init__(self, **_kwargs):
      construction_calls['task_state'] += 1

  monkeypatch.setattr(
      'google.adk.runners.ObservationWriter', _ObservationWriterStub
  )
  monkeypatch.setattr(
      'google.adk.runners.ReflectionWriter', _ReflectionWriterStub
  )
  monkeypatch.setattr(
      'google.adk.runners.TaskStateUpdater', _TaskStateUpdaterStub
  )

  first_events = await runner.run_async('first message')
  assert 'response one' in _event_texts(first_events)

  session = runner.session
  second_events = []
  async for event in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=testing_utils.UserContent('second message'),
      run_config=RunConfig(
          custom_metadata={'trigger_observational_memory_runtime': True}
      ),
  ):
    second_events.append(event)

  assert 'response two' in _event_texts(second_events)
  assert construction_calls == {
      'observation': 1,
      'reflection': 1,
      'task_state': 1,
  }
  assert compaction_service.save_observation_calls == 0
  assert compaction_service.save_reflection_calls == 0
  assert compaction_service.save_task_state_calls == 0
