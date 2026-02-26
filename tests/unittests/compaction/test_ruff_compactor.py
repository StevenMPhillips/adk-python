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

from google.adk.compaction.compactors.ruff_compactor import RuffCompactor
from google.adk.events.event import Event
from google.genai import types


def _tool_event(
    *,
    stdout: str = '',
    stderr: str = '',
    command: str = 'ruff check src',
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
      id='evt-ruff',
  )


def test_ruff_compactor_extracts_rule_codes_refs_and_messages():
  stdout = '\n'.join([
      (
          'src/google/adk/tools/mcp_tool/client.py:41:7: F401 `json` imported'
          ' but unused'
      ),
      (
          'src/google/adk/cli/cli_tools_click.py:119:13: '
          'E722 Do not use bare `except`'
      ),
      (
          'tests/unittests/agents/test_llm_agent.py:53:1: '
          'I001 Import block is un-sorted or un-formatted'
      ),
      'Found 3 errors.',
  ])

  compaction = RuffCompactor().compact(_tool_event(stdout=stdout))

  assert compaction is not None
  assert compaction.error_signatures == [
      'ruff::F401::src/google/adk/tools/mcp_tool/client.py',
      'ruff::E722::src/google/adk/cli/cli_tools_click.py',
      'ruff::I001::tests/unittests/agents/test_llm_agent.py',
  ]
  assert compaction.key_errors == [
      '`json` imported but unused',
      'Do not use bare `except`',
      'Import block is un-sorted or un-formatted',
  ]
  assert any(
      ref.path == 'src/google/adk/tools/mcp_tool/client.py'
      and ref.line == 41
      and ref.col == 7
      for ref in compaction.file_line_refs
  )
  assert any(
      ref.path == 'src/google/adk/cli/cli_tools_click.py'
      and ref.line == 119
      and ref.col == 13
      for ref in compaction.file_line_refs
  )
  assert any(
      ref.path == 'tests/unittests/agents/test_llm_agent.py'
      and ref.line == 53
      and ref.col == 1
      for ref in compaction.file_line_refs
  )


def test_ruff_compactor_applies_token_budget_to_trimmed_trace():
  stdout = '\n'.join([
      'src/google/adk/a.py:10:1: F401 item one message text',
      'src/google/adk/b.py:20:2: E501 item two message text',
      'src/google/adk/c.py:30:3: I001 item three message text',
  ])

  compaction = RuffCompactor(token_budget=16).compact(
      _tool_event(stdout=stdout)
  )

  assert compaction is not None
  assert len(compaction.trimmed_trace) < 3
  assert compaction.trimmed_trace == [] or compaction.trimmed_trace[
      0
  ].startswith('src/google/adk/a.py')
  assert compaction.error_signatures == [] or compaction.error_signatures[
      0
  ] == 'ruff::F401::src/google/adk/a.py'
  assert compaction.stats.compact_tokens_est <= 16


def test_ruff_compactor_parses_windows_and_colon_paths():
  stdout = '\n'.join([
      r'C:\repo\pkg\lint_target.py:8:3: F401 `sys` imported but unused',
      'namespace:pkg/lint_target.py:11:2: E722 Do not use bare `except`',
  ])

  compaction = RuffCompactor().compact(_tool_event(stdout=stdout))

  assert compaction is not None
  assert compaction.error_signatures == [
      r'ruff::F401::C:\repo\pkg\lint_target.py',
      'ruff::E722::namespace:pkg/lint_target.py',
  ]
  assert (r'C:\repo\pkg\lint_target.py', 8, 3) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert ('namespace:pkg/lint_target.py', 11, 2) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
