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
import re

from google.genai import types

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ..models import EvidencedItem
from ..models import EvidenceRef
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
    return len(recent_observations) >= self._observation_count_threshold

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
    generated_reflection = self._sanitize_reflection(
        reflection=generated_reflection,
        observation_ids=observation_ids,
    )
    if self._should_skip_reflection(generated_reflection):
      return None
    self._validate_reflection(
        reflection=generated_reflection,
        observation_ids=observation_ids,
    )

    reflection_to_save = generated_reflection.model_copy(
        update={'session_id': current_task_state.session_id}
    )
    await self._compaction_service.save_reflection(reflection_to_save)
    return reflection_to_save

  def _sanitize_reflection(
      self,
      *,
      reflection: Reflection,
      observation_ids: frozenset[str],
  ) -> Reflection:
    normalized_allowed = {
        observation_id.casefold(): observation_id
        for observation_id in observation_ids
    }
    covers_observation_ids = self._sanitize_observation_ids(
        reflection.covers_observation_ids,
        observation_ids=observation_ids,
        normalized_allowed=normalized_allowed,
    )
    stable_facts = self._sanitize_items(
        reflection.stable_facts,
        observation_ids=observation_ids,
        normalized_allowed=normalized_allowed,
    )
    recurring_failures = self._sanitize_items(
        reflection.recurring_failures,
        observation_ids=observation_ids,
        normalized_allowed=normalized_allowed,
    )
    strategy_updates = self._sanitize_items(
        reflection.strategy_updates,
        observation_ids=observation_ids,
        normalized_allowed=normalized_allowed,
    )
    top_level_refs = self._sanitize_evidence_refs(
        reflection.evidence_refs,
        observation_ids=observation_ids,
        normalized_allowed=normalized_allowed,
    )

    return reflection.model_copy(
        update={
            'covers_observation_ids': covers_observation_ids,
            'stable_facts': stable_facts,
            'recurring_failures': recurring_failures,
            'strategy_updates': strategy_updates,
            'evidence_refs': top_level_refs,
        }
    )

  def _should_skip_reflection(self, reflection: Reflection) -> bool:
    if reflection.covers_observation_ids:
      return False
    if reflection.stable_facts:
      return False
    if reflection.recurring_failures:
      return False
    if reflection.strategy_updates:
      return False
    return not reflection.evidence_refs

  def _sanitize_observation_ids(
      self,
      observation_id_candidates: Sequence[str],
      *,
      observation_ids: frozenset[str],
      normalized_allowed: dict[str, str],
  ) -> list[str]:
    sanitized_ids: list[str] = []
    seen_ids: set[str] = set()
    for observation_id in observation_id_candidates:
      canonical_id = self._canonicalize_observation_id(
          observation_id,
          observation_ids=observation_ids,
          normalized_allowed=normalized_allowed,
      )
      if canonical_id is None or canonical_id in seen_ids:
        continue
      seen_ids.add(canonical_id)
      sanitized_ids.append(canonical_id)
    return sanitized_ids

  def _sanitize_items(
      self,
      items: Sequence[EvidencedItem],
      *,
      observation_ids: frozenset[str],
      normalized_allowed: dict[str, str],
  ) -> list[EvidencedItem]:
    sanitized_items: list[EvidencedItem] = []
    for item in items:
      sanitized_refs = self._sanitize_evidence_refs(
          item.evidence_refs,
          observation_ids=observation_ids,
          normalized_allowed=normalized_allowed,
      )
      if not sanitized_refs:
        continue
      sanitized_items.append(item.model_copy(update={'evidence_refs': sanitized_refs}))
    return sanitized_items

  def _sanitize_evidence_refs(
      self,
      evidence_refs: Sequence[EvidenceRef],
      *,
      observation_ids: frozenset[str],
      normalized_allowed: dict[str, str],
  ) -> list[EvidenceRef]:
    sanitized_refs: list[EvidenceRef] = []
    seen_refs: set[tuple[str, str]] = set()
    for evidence_ref in evidence_refs:
      canonical_id = self._canonicalize_observation_id(
          evidence_ref.ref_id,
          observation_ids=observation_ids,
          normalized_allowed=normalized_allowed,
      )
      if canonical_id is None:
        continue
      dedupe_key = (evidence_ref.ref_type, canonical_id)
      if dedupe_key in seen_refs:
        continue
      seen_refs.add(dedupe_key)
      sanitized_refs.append(evidence_ref.model_copy(update={'ref_id': canonical_id}))
    return sanitized_refs

  def _canonicalize_observation_id(
      self,
      observation_id: str,
      *,
      observation_ids: frozenset[str],
      normalized_allowed: dict[str, str],
  ) -> str | None:
    normalized_id = observation_id.strip().strip('`"\'')
    if not normalized_id:
      return None
    if normalized_id in observation_ids:
      return normalized_id

    casefold_id = normalized_id.casefold()
    if casefold_id in normalized_allowed:
      return normalized_allowed[casefold_id]

    for token in re.split(r'[^A-Za-z0-9:_\-]+', normalized_id):
      if not token:
        continue
      if token in observation_ids:
        return token
      token_casefold = token.casefold()
      if token_casefold in normalized_allowed:
        return normalized_allowed[token_casefold]

    matching_ids: list[str] = []
    for allowed_id in observation_ids:
      match = re.search(
          rf'(^|[^A-Za-z0-9_\-]){re.escape(allowed_id)}'
          r'([^A-Za-z0-9_\-]|$)',
          normalized_id,
      )
      if match:
        matching_ids.append(allowed_id)
    if matching_ids:
      return max(matching_ids, key=len)
    return None

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
