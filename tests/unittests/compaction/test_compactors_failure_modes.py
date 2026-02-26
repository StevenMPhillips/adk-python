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

import pytest

from google.adk.compaction.compactors.generic import GenericToolRunCompactor
from google.adk.compaction.compactors.mypy_compactor import MypyCompactor
from google.adk.compaction.compactors.patch_compactor import PatchCompactor
from google.adk.compaction.compactors.pytest_compactor import PytestCompactor
from google.adk.compaction.compactors.ruff_compactor import RuffCompactor
from google.adk.events.event import Event
from google.genai import types


def _tool_event(*, response: object, event_id: str = 'evt-failure') -> Event:
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
      id=event_id,
  )


def _empty_tool_event() -> Event:
  return Event(
      author='agent',
      content=types.Content(role='user', parts=[types.Part(text='no tool output')]),
      id='evt-empty',
  )


def _file_line_ref_tuples(compaction) -> list[tuple[str, int, int | None]]:
  return [
      (ref.path, ref.line, ref.col)
      for ref in compaction.file_line_refs
  ]


@pytest.mark.parametrize(
    'compactor',
    [
        GenericToolRunCompactor(),
        RuffCompactor(),
        MypyCompactor(),
        PytestCompactor(),
        PatchCompactor(),
    ],
)
def test_compactors_return_none_for_empty_response_payload(compactor):
  compaction = compactor.compact(_tool_event(response={}))

  assert compaction is None


@pytest.mark.parametrize(
    'compactor',
    [
        GenericToolRunCompactor(),
        RuffCompactor(),
        MypyCompactor(),
        PytestCompactor(),
        PatchCompactor(),
    ],
)
def test_compactors_return_none_when_no_function_response_present(compactor):
  compaction = compactor.compact(_empty_tool_event())

  assert compaction is None


def test_generic_compactor_handles_missing_fields_with_stable_defaults():
  payload = {'stderr': ['warning one', 'warning two']}

  compaction = GenericToolRunCompactor().compact(_tool_event(response=payload))

  assert compaction is not None
  assert compaction.command == 'run_shell_command'
  assert compaction.exit_code == 0
  assert compaction.trimmed_trace == ['warning one', 'warning two']
  assert compaction.error_signatures == []


def test_ruff_compactor_preserves_signatures_and_refs_under_noisy_output():
  diag_lines = [
      'src/google/adk/a.py:10:1: F401 one unused import',
      'src/google/adk/b.py:20:2: E722 bare except',
  ]
  clean_payload = {'stdout': '\n'.join(diag_lines), 'exit_code': 1}
  noisy_payload = {
      'stdout': [
          'runner note: preparing lint',
          {'meta': 'discard-this-dict-line'},
          diag_lines[0],
          'progress 50%',
          diag_lines[1],
      ],
      'stderr': {'status': 'lint failed'},
      'exit_code': 1,
  }

  compactor = RuffCompactor()
  clean = compactor.compact(_tool_event(response=clean_payload, event_id='evt-1'))
  noisy = compactor.compact(_tool_event(response=noisy_payload, event_id='evt-2'))

  assert clean is not None
  assert noisy is not None
  assert noisy.error_signatures == clean.error_signatures
  assert noisy.key_errors == clean.key_errors
  assert _file_line_ref_tuples(noisy) == _file_line_ref_tuples(clean)


def test_mypy_compactor_preserves_signatures_and_refs_under_noisy_output():
  diag_lines = [
      (
          'src/google/adk/flows/base.py:7:3: '
          'error: Item not compatible [assignment]'
      ),
      (
          'src/google/adk/models/gemini.py:9:1: '
          'error: Missing return statement [return]'
      ),
  ]
  clean_payload = {'stdout': '\n'.join(diag_lines), 'exit_code': 1}
  noisy_payload = {
      'stdout': ['prelude line', diag_lines[0], {'debug': True}, diag_lines[1]],
      'stderr': ['mypy run metadata', {'run_id': 'abc-123'}],
      'exit_code': 1,
  }

  compactor = MypyCompactor()
  clean = compactor.compact(_tool_event(response=clean_payload, event_id='evt-3'))
  noisy = compactor.compact(_tool_event(response=noisy_payload, event_id='evt-4'))

  assert clean is not None
  assert noisy is not None
  assert noisy.error_signatures == clean.error_signatures
  assert noisy.key_errors == clean.key_errors
  assert _file_line_ref_tuples(noisy) == _file_line_ref_tuples(clean)


def test_pytest_compactor_preserves_failure_signature_under_noisy_output():
  pytest_lines = [
      'Traceback (most recent call last):',
      '  File "tests/test_auth.py", line 27, in test_login',
      '    assert login("alice", "bad")',
      'E   AssertionError: assert False',
      (
          'FAILED tests/test_auth.py::test_login '
          '- AssertionError: assert False'
      ),
  ]
  clean_payload = {'stdout': '\n'.join(pytest_lines), 'exit_code': 1}
  noisy_payload = {
      'stdout': [
          'session bootstrap',
          {'phase': 'collection'},
          *pytest_lines,
          'tail noise without refs',
      ],
      'stderr': {'stream': 'captured stderr metadata only'},
      'exit_code': 1,
  }

  compactor = PytestCompactor()
  clean = compactor.compact(_tool_event(response=clean_payload, event_id='evt-5'))
  noisy = compactor.compact(_tool_event(response=noisy_payload, event_id='evt-6'))

  assert clean is not None
  assert noisy is not None
  assert noisy.tests_failed == clean.tests_failed
  assert noisy.error_signatures == clean.error_signatures
  assert _file_line_ref_tuples(noisy) == _file_line_ref_tuples(clean)
