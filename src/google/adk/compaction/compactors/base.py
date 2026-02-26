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

import abc

from ...events.event import Event
from ..models import ToolRunCompaction


class BaseToolRunCompactor(abc.ABC):
  """Interface for tool-run compactors."""

  @abc.abstractmethod
  def compact(self, event: Event) -> ToolRunCompaction | None:
    """Compacts a tool-run event into a deterministic artifact."""
