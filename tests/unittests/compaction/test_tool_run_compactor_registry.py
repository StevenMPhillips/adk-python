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

from google.adk.compaction.compactors.base import BaseToolRunCompactor
from google.adk.compaction.compactors.generic import GenericToolRunCompactor
from google.adk.compaction.compactors.mypy_compactor import MypyCompactor
from google.adk.compaction.compactors.pytest_compactor import PytestCompactor
from google.adk.compaction.compactors.registry import ToolRunCompactorRegistry
from google.adk.compaction.compactors.ruff_compactor import RuffCompactor
from google.adk.compaction.models import ToolRunCompaction
from google.adk.events.event import Event
import pytest


class _StubCompactor(BaseToolRunCompactor):

  def compact(self, event: Event) -> ToolRunCompaction | None:
    del event
    return None


def test_registry_returns_registered_compactor_for_matching_command():
  pytest_compactor = _StubCompactor()
  mypy_compactor = _StubCompactor()
  registry = ToolRunCompactorRegistry()

  registry.register('pytest', pytest_compactor)
  registry.register('mypy', mypy_compactor)

  assert registry.get_compactor('python -m pytest -q') is pytest_compactor
  assert registry.get_compactor('MYPY src/google/adk') is mypy_compactor


def test_registry_falls_back_to_generic_compactor_for_unknown_command():
  fallback = GenericToolRunCompactor()
  registry = ToolRunCompactorRegistry(fallback=fallback)

  assert registry.get_compactor('unknown-tool --flag') is fallback


def test_registry_uses_pytest_compactor_for_pytest_commands_by_default():
  registry = ToolRunCompactorRegistry()

  assert isinstance(
      registry.get_compactor('python -m pytest -q'), PytestCompactor
  )
  assert isinstance(registry.get_compactor('/usr/bin/pytest -q'),
                    PytestCompactor)


@pytest.mark.parametrize(
    'command',
    [
        'pytest_helper --help',
        'python -m pytest_helper',
        'my_pytest_wrapper run',
    ],
)
def test_registry_does_not_match_pytest_substrings_inside_other_tokens(command):
  registry = ToolRunCompactorRegistry()

  assert isinstance(registry.get_compactor(command), GenericToolRunCompactor)


def test_registry_supports_multi_token_custom_matchers():
  custom_compactor = _StubCompactor()
  registry = ToolRunCompactorRegistry()

  registry.register('python -m pytest', custom_compactor)

  assert registry.get_compactor('python -m pytest -q tests') is custom_compactor


def test_registry_uses_mypy_compactor_for_mypy_commands_by_default():
  registry = ToolRunCompactorRegistry()

  assert isinstance(
      registry.get_compactor('mypy src/google/adk'), MypyCompactor
  )


def test_registry_uses_ruff_compactor_for_ruff_and_flake8_by_default():
  registry = ToolRunCompactorRegistry()

  assert isinstance(registry.get_compactor('ruff check src'), RuffCompactor)
  assert isinstance(registry.get_compactor('flake8 src'), RuffCompactor)


def test_registry_rejects_empty_matcher():
  registry = ToolRunCompactorRegistry()

  with pytest.raises(ValueError):
    registry.register('  ', _StubCompactor())
