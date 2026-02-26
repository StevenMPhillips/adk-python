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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.compaction.compactors.patch_compactor import PatchCompactor
from google.adk.compaction.compactors.registry import ToolRunCompactorRegistry
from google.adk.compaction.config import HybridEventsCompactionConfig
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)
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
