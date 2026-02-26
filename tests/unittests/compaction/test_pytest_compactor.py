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

from google.adk.compaction.compactors.pytest_compactor import PytestCompactor
from google.adk.events.event import Event
from google.genai import types


def _tool_event(
    *,
    stdout: str = '',
    stderr: str = '',
    command: str = 'python -m pytest tests -q',
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
      id='evt-pytest',
  )


def test_pytest_compactor_extracts_single_failure_artifacts():
  noisy_prefix = '\n'.join([f'noise line {index}' for index in range(500)])
  stdout = '\n'.join([
      noisy_prefix,
      '============================= FAILURES ==============================',
      (
          '______________________________ test_login'
          ' ______________________________'
      ),
      '    def test_login():',
      '      assert login("alice", "bad")',
      'E   AssertionError: assert False',
      '',
      'tests/test_auth.py:27: AssertionError',
      '=========================== short test summary info ===================',
      'FAILED tests/test_auth.py::test_login - AssertionError: assert False',
      '============================== 1 failed in 0.12s ======================',
  ])

  compaction = PytestCompactor().compact(_tool_event(stdout=stdout))

  assert compaction is not None
  assert compaction.tests_failed == ['tests/test_auth.py::test_login']
  assert compaction.error_signatures == [
      'pytest::AssertionError::tests/test_auth.py::test_login'
  ]
  assert 'AssertionError: assert False' in compaction.key_errors
  assert any(
      ref.path == 'tests/test_auth.py' and ref.line == 27
      for ref in compaction.file_line_refs
  )
  assert any(
      line.startswith('FAILED tests/test_auth.py::test_login')
      for line in compaction.trimmed_trace
  )
  assert compaction.stats.compression_ratio > 10.0


def test_pytest_compactor_extracts_multi_failure_traceback_artifacts():
  stdout = '\n'.join([
      '============================= FAILURES ==============================',
      '___________________________ test_fetch_user ___________________________',
      'Traceback (most recent call last):',
      '  File "/workspace/tests/test_api.py", line 44, in test_fetch_user',
      '    fetch_user("x")',
      '  File "/workspace/src/app/api.py", line 18, in fetch_user',
      '    raise ValueError("invalid user id")',
      'E   ValueError: invalid user id',
      '',
      '___________________________ test_connection ___________________________',
      'Traceback (most recent call last):',
      '  File "/workspace/tests/test_db.py", line 11, in test_connection',
      '    setup_db()',
      '  File "/workspace/src/app/db.py", line 71, in setup_db',
      '    raise RuntimeError("connection refused")',
      'E   RuntimeError: connection refused',
      '=========================== short test summary info ===================',
      'FAILED tests/test_api.py::test_fetch_user - ValueError: invalid user id',
      (
          'ERROR tests/test_db.py::test_connection '
          '- RuntimeError: connection refused'
      ),
      '========================= 1 failed, 1 error in 0.26s =================',
  ])
  stderr = '\n'.join([f'debug stderr {index}' for index in range(300)])

  compaction = PytestCompactor(token_budget=120).compact(
      _tool_event(stdout=stdout, stderr=stderr)
  )

  assert compaction is not None
  assert compaction.tests_failed == [
      'tests/test_api.py::test_fetch_user',
      'tests/test_db.py::test_connection',
  ]
  assert compaction.error_signatures == [
      'pytest::ValueError::tests/test_api.py::test_fetch_user',
      'pytest::RuntimeError::tests/test_db.py::test_connection',
  ]
  assert 'ValueError: invalid user id' in compaction.key_errors
  assert 'RuntimeError: connection refused' in compaction.key_errors
  assert any(
      ref.path == '/workspace/tests/test_api.py' and ref.line == 44
      for ref in compaction.file_line_refs
  )
  assert any(
      ref.path == '/workspace/src/app/db.py' and ref.line == 71
      for ref in compaction.file_line_refs
  )
  assert any(
      'Traceback (most recent call last):' in line
      for line in compaction.trimmed_trace
  )
  assert compaction.stats.compression_ratio > 10.0


def test_pytest_compactor_keeps_error_type_extraction_conservative():
  stdout = '\n'.join([
      'Traceback (most recent call last):',
      r'  File "C:\repo\tests\test_auth.py", line 9, in test_login',
      '    assert auth("alice")',
      'E   did not raise ValueError in helper path',
      'FAILED tests/test_auth.py::test_login - did not raise ValueError',
  ])

  compaction = PytestCompactor().compact(_tool_event(stdout=stdout))

  assert compaction is not None
  assert compaction.tests_failed == ['tests/test_auth.py::test_login']
  assert compaction.error_signatures == [
      'pytest::UnknownError::tests/test_auth.py::test_login'
  ]
  assert any(
      ref.path == r'C:\repo\tests\test_auth.py' and ref.line == 9
      for ref in compaction.file_line_refs
  )
