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

from collections.abc import Mapping
import json

from google.genai import types

from ..events.event import Event
from .models import Observation
from .models import PatchCompaction
from .models import Reflection
from .models import TaskStateAnchor
from .models import ToolRunCompaction
from .rehydration import EvidencePack
from .rehydration import RehydrationExecutor
from .rehydration import RehydrationPlanner
from .storage.base_compaction_service import BaseCompactionService

CompactionArtifact = ToolRunCompaction | PatchCompaction


class HybridPromptAssembler:
  """Builds hybrid prompt layers from compaction artifacts and raw tail."""

  def __init__(
      self,
      *,
      compaction_service: BaseCompactionService,
      prompt_token_budget: int,
      raw_turns_count: int,
      compactions_count: int,
      observations_count: int,
      rehydration_evidence_token_budget: int,
      rehydration_planner: RehydrationPlanner | None = None,
  ):
    self._compaction_service = compaction_service
    self._prompt_token_budget = max(1, prompt_token_budget)
    self._raw_turns_count = max(1, raw_turns_count)
    self._compactions_count = max(1, compactions_count)
    self._observations_count = min(3, max(1, observations_count))
    self._rehydration_evidence_token_budget = max(
        1, rehydration_evidence_token_budget
    )
    self._rehydration_planner = rehydration_planner or RehydrationPlanner()

  async def assemble(
      self,
      *,
      session_id: str,
      events: list[Event],
      baseline_contents: list[types.Content],
      system_instruction: str | None,
  ) -> list[types.Content]:
    """Assembles a layered prompt within the configured token budget."""
    remaining_tokens = self._prompt_token_budget - _estimate_tokens(
        system_instruction or ''
    )
    if remaining_tokens <= 0:
      return baseline_contents

    task_state = await self._compaction_service.get_task_state(session_id)
    latest_reflection = await self._compaction_service.get_latest_reflection(
        session_id
    )
    recent_observations = await self._compaction_service.get_observations(
        session_id
    )
    selected_observations = recent_observations[-self._observations_count :]

    latest_compactions = await self._latest_compactions_from_events(events)
    latest_user_message = _latest_user_message(events)
    evidence_pack = await self._maybe_build_evidence_pack(
        task_state=task_state,
        observations=selected_observations,
        recent_compactions=latest_compactions,
        latest_user_message=latest_user_message,
        event_time_by_id={event.id: i + 1 for i, event in enumerate(events)},
    )

    prompt_contents: list[types.Content] = []

    remaining_tokens = _append_text_layer(
        prompt_contents=prompt_contents,
        remaining_tokens=remaining_tokens,
        title='TaskState',
        payload=_task_state_payload(task_state),
    )
    remaining_tokens = _append_text_layer(
        prompt_contents=prompt_contents,
        remaining_tokens=remaining_tokens,
        title='Reflection',
        payload=_reflection_payload(latest_reflection),
    )
    remaining_tokens = _append_text_layer(
        prompt_contents=prompt_contents,
        remaining_tokens=remaining_tokens,
        title='Observations',
        payload=_observations_payload(selected_observations),
    )
    remaining_tokens = _append_text_layer(
        prompt_contents=prompt_contents,
        remaining_tokens=remaining_tokens,
        title='Evidence Pack',
        payload=_evidence_pack_payload(evidence_pack),
    )

    remaining_tokens, raw_tail = _select_raw_tail_contents(
        baseline_contents=baseline_contents,
        raw_turns_count=self._raw_turns_count,
        remaining_tokens=remaining_tokens,
    )
    prompt_contents.extend(raw_tail)

    selected_compactions = _select_compactions_within_budget(
        compactions=latest_compactions,
        max_compactions=self._compactions_count,
        remaining_tokens=remaining_tokens,
    )
    _append_text_layer(
        prompt_contents=prompt_contents,
        remaining_tokens=remaining_tokens,
        title='Recent Compactions',
        payload=_compactions_payload(selected_compactions),
    )

    return prompt_contents or baseline_contents

  async def _latest_compactions_from_events(
      self, events: list[Event]
  ) -> list[CompactionArtifact]:
    event_ids: list[str] = []
    for event in reversed(events):
      if event.get_function_responses():
        event_ids.append(event.id)

    artifacts: list[CompactionArtifact] = []
    for event_id in event_ids:
      tool_run = await self._compaction_service.get_tool_run_compaction(
          event_id
      )
      if tool_run is not None:
        artifacts.append(tool_run)
      patch = await self._compaction_service.get_patch_compaction(event_id)
      if patch is not None:
        artifacts.append(patch)
      if len(artifacts) >= self._compactions_count:
        break
    return artifacts

  async def _maybe_build_evidence_pack(
      self,
      *,
      task_state: TaskStateAnchor | None,
      observations: list[Observation],
      recent_compactions: list[CompactionArtifact],
      latest_user_message: str,
      event_time_by_id: Mapping[str, int],
  ) -> EvidencePack | None:
    if task_state is None:
      return None

    queries = self._rehydration_planner.plan(
        task_state=task_state,
        recent_observations=observations,
        recent_compactions=recent_compactions,
        latest_user_message=latest_user_message,
    )
    if not queries:
      return None

    executor = RehydrationExecutor(
        self._compaction_service,
        token_budget=self._rehydration_evidence_token_budget,
    )
    evidence_pack = await executor.execute(
        queries,
        event_time_by_id=event_time_by_id,
    )
    if not evidence_pack.compacted_artifacts and not evidence_pack.raw_excerpts:
      return None
    return evidence_pack


def _select_raw_tail_contents(
    *,
    baseline_contents: list[types.Content],
    raw_turns_count: int,
    remaining_tokens: int,
) -> tuple[int, list[types.Content]]:
  if remaining_tokens <= 0:
    return remaining_tokens, []

  selected: list[types.Content] = []
  for content in reversed(baseline_contents[-raw_turns_count:]):
    content_tokens = _estimate_content_tokens(content)
    if content_tokens > remaining_tokens:
      continue
    selected.append(content)
    remaining_tokens -= content_tokens
  selected.reverse()
  return remaining_tokens, selected


def _select_compactions_within_budget(
    *,
    compactions: list[CompactionArtifact],
    max_compactions: int,
    remaining_tokens: int,
) -> list[CompactionArtifact]:
  if remaining_tokens <= 0:
    return []

  selected: list[CompactionArtifact] = []
  for compaction in compactions[:max_compactions]:
    compact_tokens = compaction.stats.compact_tokens_est
    if compact_tokens > remaining_tokens:
      continue
    selected.append(compaction)
    remaining_tokens -= compact_tokens
  return selected


def _append_text_layer(
    *,
    prompt_contents: list[types.Content],
    remaining_tokens: int,
    title: str,
    payload: object | None,
) -> int:
  if payload is None:
    return remaining_tokens

  payload_text = json.dumps(payload, indent=2, sort_keys=True)
  layer_text = f'[{title}]\n{payload_text}'
  layer_tokens = _estimate_tokens(layer_text)
  if layer_tokens > remaining_tokens:
    return remaining_tokens

  prompt_contents.append(
      types.Content(role='user', parts=[types.Part(text=layer_text)])
  )
  return remaining_tokens - layer_tokens


def _task_state_payload(task_state: TaskStateAnchor | None) -> object | None:
  if task_state is None:
    return None
  return task_state.model_dump(mode='json', by_alias=True)


def _reflection_payload(reflection: Reflection | None) -> object | None:
  if reflection is None:
    return None
  return reflection.model_dump(mode='json', by_alias=True)


def _observations_payload(observations: list[Observation]) -> object | None:
  if not observations:
    return None
  return [obs.model_dump(mode='json', by_alias=True) for obs in observations]


def _evidence_pack_payload(evidence_pack: EvidencePack | None) -> object | None:
  if evidence_pack is None:
    return None
  return evidence_pack.model_dump(mode='json', by_alias=True)


def _compactions_payload(
    compactions: list[CompactionArtifact],
) -> object | None:
  if not compactions:
    return None
  return [
      compaction.model_dump(mode='json', by_alias=True)
      for compaction in compactions
  ]


def _latest_user_message(events: list[Event]) -> str:
  for event in reversed(events):
    if event.author != 'user' or not event.content or not event.content.parts:
      continue
    text = ''.join(part.text or '' for part in event.content.parts)
    if text:
      return text
  return ''


def _estimate_content_tokens(content: types.Content) -> int:
  text_parts: list[str] = []
  for part in content.parts:
    if part.text:
      text_parts.append(part.text)
    if part.function_call:
      text_parts.append(part.function_call.name)
      text_parts.append(json.dumps(part.function_call.args, sort_keys=True))
    if part.function_response:
      text_parts.append(part.function_response.name)
      text_parts.append(
          json.dumps(part.function_response.response, sort_keys=True)
      )
  return _estimate_tokens(''.join(text_parts))


def _estimate_tokens(text: str) -> int:
  if not text:
    return 0
  return max(1, len(text) // 4)
