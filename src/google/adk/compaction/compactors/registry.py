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

import shlex

from .base import BaseToolRunCompactor
from .generic import GenericToolRunCompactor
from .mypy_compactor import MypyCompactor
from .pytest_compactor import PytestCompactor
from .ruff_compactor import RuffCompactor


class ToolRunCompactorRegistry:
  """Registry for selecting tool-run compactors by command match."""

  def __init__(self, fallback: BaseToolRunCompactor | None = None):
    self._fallback = fallback or GenericToolRunCompactor()
    ruff_compactor = RuffCompactor()
    self._registrations: list[
        tuple[tuple[str, ...], BaseToolRunCompactor]
    ] = [
        (self._normalize_matcher('pytest'), PytestCompactor()),
        (self._normalize_matcher('mypy'), MypyCompactor()),
        (self._normalize_matcher('ruff'), ruff_compactor),
        (self._normalize_matcher('flake8'), ruff_compactor),
    ]

  def register(
      self, command_match: str, compactor: BaseToolRunCompactor
  ) -> None:
    """Registers a compactor for token-aware command matching."""
    normalized = self._normalize_matcher(command_match)
    if not normalized:
      raise ValueError('command_match must be non-empty.')
    self._registrations.append((normalized, compactor))

  def get_compactor(self, command: str) -> BaseToolRunCompactor:
    """Returns the first registered compactor matching command tokens."""
    command_tokens = self._tokenize(command)
    for matcher, compactor in reversed(self._registrations):
      if self._matcher_in_command(matcher, command_tokens):
        return compactor
    return self._fallback

  @staticmethod
  def _normalize_matcher(command_match: str) -> tuple[str, ...]:
    return tuple(ToolRunCompactorRegistry._tokenize(command_match))

  @staticmethod
  def _tokenize(command: str) -> list[str]:
    try:
      tokens = shlex.split(command)
    except ValueError:
      tokens = command.split()
    return [token.lower() for token in tokens]

  @staticmethod
  def _matcher_in_command(
      matcher_tokens: tuple[str, ...], command_tokens: list[str]
  ) -> bool:
    if not matcher_tokens or len(matcher_tokens) > len(command_tokens):
      return False
    for start in range(len(command_tokens) - len(matcher_tokens) + 1):
      if all(
          ToolRunCompactorRegistry._token_matches(matcher_token, command_token)
          for matcher_token, command_token in zip(
              matcher_tokens,
              command_tokens[start : start + len(matcher_tokens)],
          )
      ):
        return True
    return False

  @staticmethod
  def _token_matches(matcher_token: str, command_token: str) -> bool:
    if matcher_token == command_token:
      return True
    return (
        command_token.rsplit('/', maxsplit=1)[-1] == matcher_token
        or command_token.rsplit('\\', maxsplit=1)[-1] == matcher_token
    )
