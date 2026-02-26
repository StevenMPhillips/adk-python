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

from collections.abc import Sequence
import json

from google.genai import types

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ..models import Observation
from ..models import Reflection
from ..models import TaskStateAnchor
from ..storage.base_compaction_service import BaseCompactionService


class ReflectionWriter:
  """Generates and persists a reflection from recent observations."""

  _DEFAULT_OBSERVATION_COUNT_THRESHOLD = 10
  _DEFAULT_MAX_OBSERVATION_WINDOW = 20

  def __init__(
      self,
      *,
      llm: BaseLlm,
      compaction_service: BaseCompactionService,
      observation_count_threshold: int = _DEFAULT_OBSERVATION_COUNT_THRESHOLD,
      max_observation_window: int = _DEFAULT_MAX_OBSERVATION_WINDOW,
  ):
    if observation_count_threshold <= 0:
      raise ValueError('observation_count_threshold must be greater than zero.')
    if max_observation_window <= 0:
      raise ValueError('max_observation_window must be greater than zero.')
    if max_observation_window < observation_count_threshold:
      raise ValueError(
          'max_observation_window must be at least observation_count_threshold.'
      )

    self._llm = llm
    self._compaction_service = compaction_service
    self._observation_count_threshold = observation_count_threshold
    self._max_observation_window = max_observation_window

  def should_write_reflection(
      self, *, recent_observations: Sequence[Observation]
  ) -> bool:
    """Returns whether reflection generation should run for this window."""
    return len(recent_observations) > self._observation_count_threshold

  async def maybe_write_reflection(
      self,
      *,
      recent_observations: Sequence[Observation],
      current_task_state: TaskStateAnchor,
      latest_reflection: Reflection | None = None,
  ) -> Reflection | None:
    """Writes one reflection when trigger conditions are met."""
    if not self.should_write_reflection(
        recent_observations=recent_observations,
    ):
      return None

    self._validate_inputs(
        recent_observations=recent_observations,
        current_task_state=current_task_state,
    )
    observation_window = self._select_observation_window(recent_observations)
    observation_ids = frozenset(
        observation.observation_id for observation in observation_window
    )

    prompt = self._build_prompt(
        observation_window=observation_window,
        current_task_state=current_task_state,
        latest_reflection=latest_reflection,
    )
    llm_request = LlmRequest(
        model=self._llm.model,
        contents=[types.Content(role='user', parts=[types.Part(text=prompt)])],
    )
    llm_request.set_output_schema(Reflection)
    raw_json = await self._generate_json_output(llm_request)
    generated_reflection = Reflection.model_validate_json(raw_json)
    self._validate_reflection(
        reflection=generated_reflection,
        observation_ids=observation_ids,
    )

    reflection_to_save = generated_reflection.model_copy(
        update={'session_id': current_task_state.session_id}
    )
    await self._compaction_service.save_reflection(reflection_to_save)
    return reflection_to_save

  async def _generate_json_output(self, llm_request: LlmRequest) -> str:
    """Runs the LLM and extracts a JSON object from text parts."""
    async for llm_response in self._llm.generate_content_async(
        llm_request, stream=False
    ):
      if not llm_response.content or not llm_response.content.parts:
        continue
      response_text = ''.join(
          part.text for part in llm_response.content.parts if part.text
      )
      if response_text:
        return response_text
    raise ValueError('ReflectionWriter LLM response did not include JSON text.')

  def _validate_inputs(
      self,
      *,
      recent_observations: Sequence[Observation],
      current_task_state: TaskStateAnchor,
  ) -> None:
    if not recent_observations:
      raise ValueError('recent_observations must not be empty.')
    session_ids = {
        observation.session_id for observation in recent_observations
    }
    if len(session_ids) != 1:
      raise ValueError('All recent_observations must belong to one session.')
    session_id = next(iter(session_ids))
    if session_id != current_task_state.session_id:
      raise ValueError('Task state session does not match recent_observations.')

  def _select_observation_window(
      self, recent_observations: Sequence[Observation]
  ) -> list[Observation]:
    ordered_observations = sorted(
        recent_observations,
        key=lambda observation: (observation.end_seq, observation.start_seq),
    )
    return ordered_observations[-self._max_observation_window :]

  def _build_prompt(
      self,
      *,
      observation_window: Sequence[Observation],
      current_task_state: TaskStateAnchor,
      latest_reflection: Reflection | None,
  ) -> str:
    payload = {
        'sessionId': current_task_state.session_id,
        'observationWindow': [
            observation.model_dump(mode='json', by_alias=True)
            for observation in observation_window
        ],
        'currentTaskState': current_task_state.model_dump(
            mode='json', by_alias=True
        ),
        'latestReflection': (
            None
            if latest_reflection is None
            else latest_reflection.model_dump(mode='json', by_alias=True)
        ),
        'allowedObservationIds': sorted(
            observation.observation_id for observation in observation_window
        ),
    }
    context_json = json.dumps(payload, indent=2, sort_keys=True)
    return (
        'Generate exactly one Reflection JSON object based on the '
        'observationWindow. coversObservationIds must include one or more '
        'observation IDs from allowedObservationIds. Every item in '
        'stableFacts, '
        'recurringFailures, and strategyUpdates must include at least one '
        'evidenceRefs entry. Every evidenceRefs.refId must be in '
        'allowedObservationIds. Do not introduce unsupported facts. Omit '
        'unsupported claims, or label uncertain claims as hypotheses. Prefix '
        'hypothesis text with "Hypothesis:". Return JSON only.\n\n'
        f'Context:\n{context_json}'
    )

  def _validate_reflection(
      self,
      *,
      reflection: Reflection,
      observation_ids: frozenset[str],
  ) -> None:
    if not reflection.covers_observation_ids:
      raise ValueError('covers_observation_ids must include at least one id.')
    for observation_id in reflection.covers_observation_ids:
      if observation_id not in observation_ids:
        raise ValueError(
            'covers_observation_ids must reference allowed observation ids.'
        )

    supported_items = (
        reflection.stable_facts
        + reflection.recurring_failures
        + reflection.strategy_updates
    )
    for item in supported_items:
      if not item.evidence_refs:
        raise ValueError(
            'Each stable/recurring/strategy item must include evidence_refs.'
        )
      for evidence_ref in item.evidence_refs:
        if evidence_ref.ref_id not in observation_ids:
          raise ValueError(
              'Reflection evidence ref id is not in allowed observation ids.'
          )

    for evidence_ref in reflection.evidence_refs:
      if evidence_ref.ref_id not in observation_ids:
        raise ValueError(
            'Top-level evidence ref id is not in allowed observation ids.'
        )
