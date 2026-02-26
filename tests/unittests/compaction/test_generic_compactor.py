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

from google.adk.compaction.compactors.generic import GenericToolRunCompactor
from google.adk.events.event import Event
from google.genai import types


def _tool_event(response: dict[str, object]) -> Event:
  return Event(
      author='agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='run_shell_command',
                      response=response,
                  )
              )
          ],
      ),
      id='evt-tool-run',
  )


def test_generic_compactor_extracts_file_line_refs_and_trims_stderr_tail():
  stderr_lines = [f'stderr line {i}' for i in range(1, 26)]
  stderr_lines[4] = 'src/google/adk/flows/llm_flow.py:17: RuntimeError'
  stderr_lines[20] = (
      'tests/unittests/compaction/test_generic_compactor.py:42:7: '
      'AssertionError'
  )
  response = {
      'command': 'python -m pytest tests/unittests/compaction -q',
      'exit_code': 1,
      'stdout': 'Ran tests\n',
      'stderr': '\n'.join(stderr_lines),
  }

  compactor = GenericToolRunCompactor(stderr_tail_lines=20, token_budget=600)
  compaction = compactor.compact(_tool_event(response=response))

  assert compaction is not None
  assert compaction.exit_code == 1
  assert len(compaction.trimmed_trace) == 20
  assert compaction.trimmed_trace[0] == 'stderr line 6'
  assert compaction.trimmed_trace[-1] == 'stderr line 25'
  assert compaction.file_line_refs[0].path == 'src/google/adk/flows/llm_flow.py'
  assert compaction.file_line_refs[0].line == 17
  assert compaction.file_line_refs[0].col is None
  assert (
      compaction.file_line_refs[1].path
      == 'tests/unittests/compaction/test_generic_compactor.py'
  )
  assert compaction.file_line_refs[1].line == 42
  assert compaction.file_line_refs[1].col == 7
  assert compaction.stats.raw_tokens_est >= compaction.stats.compact_tokens_est


def test_generic_compactor_applies_token_budget_cap_on_trimmed_trace():
  response: dict[str, object] = {
      'command': 'ruff check src',
      'exitCode': '2',
      'stderr': '\n'.join(['1234567890', 'abcdefghij', 'XYZ']),
  }

  compactor = GenericToolRunCompactor(stderr_tail_lines=20, token_budget=5)
  compaction = compactor.compact(_tool_event(response=response))

  assert compaction is not None
  assert compaction.exit_code == 2
  assert compaction.trimmed_trace == ['abcdefghij', 'XYZ']
  assert compaction.stats.compact_tokens_est <= 5


def test_generic_compactor_handles_cross_platform_path_edge_cases():
  stderr = '\n'.join([
      r'C:\repo\pkg-dir\module.py:10:2: RuntimeError',
      r'\\server\share\lib\node.py:22:1: LookupError',
      '.env:7: bad value',
      './configs/.pre-commit-config.yaml:3: malformed',
      'docs/my-file-name:9: missing heading',
      'elapsed 12:34',
  ])
  response: dict[str, object] = {
      'command': 'python script.py',
      'exit_code': 1,
      'stderr': stderr,
  }

  compaction = GenericToolRunCompactor().compact(_tool_event(response=response))

  assert compaction is not None
  assert (r'C:\repo\pkg-dir\module.py', 10, 2) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert (r'\\server\share\lib\node.py', 22, 1) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert ('.env', 7, None) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert ('./configs/.pre-commit-config.yaml', 3, None) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert ('docs/my-file-name', 9, None) in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
  assert ('12', 34, None) not in {
      (ref.path, ref.line, ref.col) for ref in compaction.file_line_refs
  }
