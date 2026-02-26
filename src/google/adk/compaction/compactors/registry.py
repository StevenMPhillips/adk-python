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

from .base import BaseToolRunCompactor
from .generic import GenericToolRunCompactor
from .pytest_compactor import PytestCompactor


class ToolRunCompactorRegistry:
  """Registry for selecting tool-run compactors by command match."""

  def __init__(self, fallback: BaseToolRunCompactor | None = None):
    self._fallback = fallback or GenericToolRunCompactor()
    self._registrations: list[tuple[str, BaseToolRunCompactor]] = [
        ('pytest', PytestCompactor())
    ]

  def register(self, command_match: str, compactor: BaseToolRunCompactor) -> None:
    """Registers a compactor for command substring matching."""
    normalized = command_match.strip().lower()
    if not normalized:
      raise ValueError('command_match must be non-empty.')
    self._registrations.append((normalized, compactor))

  def get_compactor(self, command: str) -> BaseToolRunCompactor:
    """Returns the first registered compactor matching the command text."""
    normalized = command.lower()
    for matcher, compactor in reversed(self._registrations):
      if matcher in normalized:
        return compactor
    return self._fallback
