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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
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
  if command.startswith('pytest'):
    return {
        'command': command,
        'exit_code': 1,
        'stderr': (
            'tests/unittests/runners/test_hybrid_compaction.py:99: '
            'AssertionError'
        ),
        'stdout': '',
        'duration_ms': 27,
    }

  patch = '\n'.join([
      'diff --git a/src/demo.py b/src/demo.py',
      '--- a/src/demo.py',
      '+++ b/src/demo.py',
      '@@ -1,1 +1,1 @@',
      '-print("old")',
      '+print("new")',
  ])
  return {
      'command': command,
      'exit_code': 0,
      'diff': patch,
      'stdout': patch,
  }


@pytest.mark.asyncio
async def test_runner_creates_deterministic_artifacts_for_multi_turn_tool_runs():
  tool = FunctionTool(func=_run_shell)
  llm_responses = [
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name,
                          args={
                              'command': (
                                  'pytest tests/unittests/runners/'
                                  'test_hybrid_compaction.py'
                              )
                          },
                      )
                  )
              ]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[Part(text='turn 1 complete')]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name,
                          args={'command': 'git diff -- src/demo.py'},
                      )
                  )
              ]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[Part(text='turn 2 complete')]
          )
      ),
  ]
  model = testing_utils.MockModel.create(llm_responses)
  compaction_service = InMemoryCompactionService()
  app = App(
      name='hybrid_compaction_app',
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

  await runner.run_async('run tests')
  await runner.run_async('show me the patch')

  tool_response_events = [
      event for event in runner.session.events if event.get_function_responses()
  ]
  assert len(tool_response_events) == 2

  first_tool_run = await compaction_service.get_tool_run_compaction(
      tool_response_events[0].id
  )
  second_tool_run = await compaction_service.get_tool_run_compaction(
      tool_response_events[1].id
  )
  first_patch = await compaction_service.get_patch_compaction(
      tool_response_events[0].id
  )
  second_patch = await compaction_service.get_patch_compaction(
      tool_response_events[1].id
  )

  assert first_tool_run is not None
  assert second_tool_run is not None
  assert first_tool_run.command.startswith('pytest')
  assert second_tool_run.command.startswith('git diff')
  assert first_patch is None
  assert second_patch is not None
  assert second_patch.files_changed == ['src/demo.py']
