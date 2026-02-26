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

from google.adk.compaction.compactors.mypy_compactor import MypyCompactor
from google.adk.events.event import Event
from google.genai import types


def _tool_event(
    *,
    stdout: str = '',
    stderr: str = '',
    command: str = 'mypy src/google/adk',
) -> Event:
  return Event(
      author='agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='run_shell_command',
                      response={
                          'command': command,
                          'stdout': stdout,
                          'stderr': stderr,
                          'exit_code': 1,
                      },
                  )
              )
          ],
      ),
      id='evt-mypy',
  )


def test_mypy_compactor_extracts_error_codes_refs_and_messages():
  stdout = '\n'.join([
      (
          'src/google/adk/flows/llm_flows/base_llm_flow.py:127:9: '
          'error: Incompatible types in assignment '
          '[assignment]'
      ),
      (
          'src/google/adk/models/gemini_llm_connection.py:88:5: '
          'error: "None" has no attribute "send" '
          '[union-attr]'
      ),
      (
          'tests/unittests/streaming/test_live.py:61:13: '
          'error: Missing return statement [return]'
      ),
      'Found 3 errors in 3 files (checked 412 source files)',
  ])

  compaction = MypyCompactor().compact(_tool_event(stdout=stdout))

  assert compaction is not None
  assert compaction.error_signatures == [
      'mypy::assignment::src/google/adk/flows/llm_flows/base_llm_flow.py',
      'mypy::union-attr::src/google/adk/models/gemini_llm_connection.py',
      'mypy::return::tests/unittests/streaming/test_live.py',
  ]
  assert compaction.key_errors == [
      'Incompatible types in assignment',
      '"None" has no attribute "send"',
      'Missing return statement',
  ]
  assert any(
      ref.path == 'src/google/adk/flows/llm_flows/base_llm_flow.py'
      and ref.line == 127
      and ref.col == 9
      for ref in compaction.file_line_refs
  )
  assert any(
      ref.path == 'src/google/adk/models/gemini_llm_connection.py'
      and ref.line == 88
      and ref.col == 5
      for ref in compaction.file_line_refs
  )
  assert any(
      ref.path == 'tests/unittests/streaming/test_live.py'
      and ref.line == 61
      and ref.col == 13
      for ref in compaction.file_line_refs
  )


def test_mypy_compactor_applies_token_budget_to_trimmed_trace():
  stdout = '\n'.join([
      'src/google/adk/agents/agent.py:10:1: error: Item one message [misc]',
      (
          'src/google/adk/agents/runner.py:22:2: '
          'error: Item two message [arg-type]'
      ),
      (
          'src/google/adk/agents/state.py:33:3: '
          'error: Item three message [assignment]'
      ),
  ])

  compaction = MypyCompactor(token_budget=20).compact(
      _tool_event(stdout=stdout)
  )

  assert compaction is not None
  assert len(compaction.trimmed_trace) < 3
  assert compaction.trimmed_trace == [] or compaction.trimmed_trace[
      0
  ].startswith('src/google/adk/agents/agent.py')
  assert compaction.error_signatures == [] or compaction.error_signatures[
      0
  ] == ('mypy::misc::src/google/adk/agents/agent.py')
  assert compaction.stats.compact_tokens_est <= 20


def test_mypy_compactor_parses_windows_and_colon_paths():
  stdout = '\n'.join([
      (
          r'C:\repo\src\app\service.py:15:4: '
          r'error: Name "token" is not defined [name-defined]'
      ),
      (
          'namespace:pkg/module.py:21:9: '
          'error: Incompatible return value type [return-value]'
      ),
  ])

  compaction = MypyCompactor().compact(_tool_event(stdout=stdout))

  assert compaction is not None
  assert compaction.error_signatures == [
      r'mypy::name-defined::C:\repo\src\app\service.py',
      'mypy::return-value::namespace:pkg/module.py',
  ]
  assert (r'C:\repo\src\app\service.py', 15, 4) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert ('namespace:pkg/module.py', 21, 9) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
