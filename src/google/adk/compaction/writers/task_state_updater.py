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
from ..models import EvidencedItem
from ..models import EvidenceRef
from ..models import Observation
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction
from ..storage.base_compaction_service import BaseCompactionService


class TaskStateUpdater:
  """Updates and persists compact task-state anchors."""

  _DEFAULT_TASK_STATE_TOKEN_BUDGET = 200

  def __init__(
      self,
      *,
      compaction_service: BaseCompactionService,
      llm: BaseLlm | None = None,
      task_state_token_budget: int = _DEFAULT_TASK_STATE_TOKEN_BUDGET,
  ):
    if task_state_token_budget <= 0:
      raise ValueError('task_state_token_budget must be greater than zero.')
    self._compaction_service = compaction_service
    self._llm = llm
    self._task_state_token_budget = task_state_token_budget

  async def update_from_observation(
      self,
      *,
      observation: Observation,
      current_task_state: TaskStateAnchor,
      explicit_user_instructions: Sequence[str] = (),
      use_llm: bool = False,
  ) -> TaskStateAnchor:
    """Updates task state from one observation and persists it."""
    if observation.session_id != current_task_state.session_id:
      raise ValueError(
          'Observation session_id must match current task-state session_id.'
      )

    explicit_instruction_texts = self._normalized_texts(
        explicit_user_instructions
    )
    inferred_decision_texts = self._inferred_decision_texts(observation)
    allowed_evidence_ref_ids = self._observation_evidence_ref_ids(observation)

    if use_llm:
      candidate_state = await self._generate_state_with_llm(
          observation=observation,
          current_task_state=current_task_state,
          explicit_user_instructions=explicit_user_instructions,
          allowed_evidence_ref_ids=allowed_evidence_ref_ids,
          inferred_decision_texts=inferred_decision_texts,
      )
    else:
      candidate_state = self._update_from_observation_deterministically(
          observation=observation,
          current_task_state=current_task_state,
      )

    updated_state = self._finalize_state(
        candidate_state=candidate_state,
        current_task_state=current_task_state,
        explicit_instruction_texts=explicit_instruction_texts,
        inferred_decision_texts=inferred_decision_texts,
        allowed_evidence_ref_ids=allowed_evidence_ref_ids,
        last_updated_seq=observation.end_seq,
    )
    await self._compaction_service.save_task_state(updated_state)
    return updated_state

  async def update_from_tool_run(
      self,
      *,
      tool_run_compaction: ToolRunCompaction,
      current_task_state: TaskStateAnchor,
      explicit_user_instructions: Sequence[str] = (),
  ) -> TaskStateAnchor:
    """Updates task state from deterministic tool-run compaction output."""
    evidence_ref = EvidenceRef(
        ref_type='tool_run',
        ref_id=tool_run_compaction.event_id,
    )
    added_known_failures = [
        EvidencedItem(
            text=f'Error signature: {signature}',
            evidence_refs=[evidence_ref],
        )
        for signature in tool_run_compaction.error_signatures
    ]
    added_known_failures.extend([
        EvidencedItem(
            text=f'Failed test: {test_id}',
            evidence_refs=[evidence_ref],
        )
        for test_id in tool_run_compaction.tests_failed
    ])
    added_hypotheses = [
        EvidencedItem(text=error_text, evidence_refs=[evidence_ref])
        for error_text in tool_run_compaction.key_errors
    ]

    candidate_state = current_task_state.model_copy(deep=True)
    candidate_state.known_failures = self._merge_items(
        candidate_state.known_failures,
        added_known_failures,
    )
    candidate_state.hypotheses = self._merge_items(
        candidate_state.hypotheses,
        added_hypotheses,
    )
    if explicit_user_instructions:
      candidate_state.constraints = self._merge_items(
          candidate_state.constraints,
          [
              EvidencedItem(text=text, evidence_refs=[])
              for text in explicit_user_instructions
          ],
      )

    updated_state = self._finalize_state(
        candidate_state=candidate_state,
        current_task_state=current_task_state,
        explicit_instruction_texts=self._normalized_texts(
            explicit_user_instructions
        ),
        inferred_decision_texts=frozenset(),
        allowed_evidence_ref_ids=frozenset([tool_run_compaction.event_id]),
        last_updated_seq=current_task_state.last_updated_seq,
    )
    await self._compaction_service.save_task_state(updated_state)
    return updated_state

  async def _generate_state_with_llm(
      self,
      *,
      observation: Observation,
      current_task_state: TaskStateAnchor,
      explicit_user_instructions: Sequence[str],
      allowed_evidence_ref_ids: frozenset[str],
      inferred_decision_texts: frozenset[str],
  ) -> TaskStateAnchor:
    if self._llm is None:
      raise ValueError('TaskStateUpdater requires llm when use_llm=True.')

    context_payload = {
        'observation': observation.model_dump(mode='json', by_alias=True),
        'currentTaskState': current_task_state.model_dump(
            mode='json', by_alias=True
        ),
        'explicitUserInstructions': list(explicit_user_instructions),
        'allowedEvidenceRefIds': sorted(allowed_evidence_ref_ids),
        'inferredDecisionTexts': sorted(inferred_decision_texts),
    }
    context_json = json.dumps(context_payload, sort_keys=True, indent=2)
    prompt = (
        'Generate exactly one TaskStateAnchor JSON object. Keep constraints '
        'small and stable. A constraint is valid only when it has '
        'evidenceRefs with refId in allowedEvidenceRefIds or exactly matches '
        'an explicitUserInstructions entry. Statements in '
        'inferredDecisionTexts must go to hypotheses, never constraints. Keep '
        'the result concise. Return JSON only.\n\n'
        f'Context:\n{context_json}'
    )

    llm_request = LlmRequest(
        model=self._llm.model,
        contents=[types.Content(role='user', parts=[types.Part(text=prompt)])],
    )
    llm_request.set_output_schema(TaskStateAnchor)
    raw_json = await self._generate_json_output(llm_request)
    return TaskStateAnchor.model_validate_json(raw_json)

  async def _generate_json_output(self, llm_request: LlmRequest) -> str:
    """Runs the LLM and extracts a JSON object from text parts."""
    if self._llm is None:
      raise ValueError('TaskStateUpdater requires llm for JSON generation.')

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
    raise ValueError('TaskStateUpdater LLM response did not include JSON text.')

  def _update_from_observation_deterministically(
      self,
      *,
      observation: Observation,
      current_task_state: TaskStateAnchor,
  ) -> TaskStateAnchor:
    candidate_state = current_task_state.model_copy(deep=True)
    explicit_items: list[EvidencedItem] = []
    inferred_items: list[EvidencedItem] = []

    for decision in observation.decisions + observation.learned_constraints:
      item = EvidencedItem(
          text=decision.text,
          evidence_refs=decision.evidence_refs,
      )
      if decision.kind == 'inferred':
        inferred_items.append(item)
      else:
        explicit_items.append(item)

    candidate_state.constraints = self._merge_items(
        candidate_state.constraints,
        explicit_items,
    )
    candidate_state.hypotheses = self._merge_items(
        candidate_state.hypotheses,
        inferred_items,
    )
    candidate_state.next_steps = self._merge_items(
        candidate_state.next_steps,
        observation.next_steps,
    )
    return candidate_state

  def _finalize_state(
      self,
      *,
      candidate_state: TaskStateAnchor,
      current_task_state: TaskStateAnchor,
      explicit_instruction_texts: frozenset[str],
      inferred_decision_texts: frozenset[str],
      allowed_evidence_ref_ids: frozenset[str],
      last_updated_seq: int,
  ) -> TaskStateAnchor:
    existing_constraint_texts = {
        self._normalize_text(item.text)
        for item in current_task_state.constraints
    }
    self._validate_constraint_provenance(
        constraints=candidate_state.constraints,
        explicit_instruction_texts=explicit_instruction_texts,
        inferred_decision_texts=inferred_decision_texts,
        allowed_evidence_ref_ids=allowed_evidence_ref_ids,
        existing_constraint_texts=existing_constraint_texts,
    )

    normalized_objective = candidate_state.objective.strip()
    if not normalized_objective:
      normalized_objective = current_task_state.objective

    updated_state = candidate_state.model_copy(
        update={
            'session_id': current_task_state.session_id,
            'objective': normalized_objective,
            'state_version': current_task_state.state_version + 1,
            'last_updated_seq': max(
                current_task_state.last_updated_seq,
                last_updated_seq,
            ),
        },
        deep=True,
    )
    return self._apply_token_budget(updated_state)

  def _validate_constraint_provenance(
      self,
      *,
      constraints: Sequence[EvidencedItem],
      explicit_instruction_texts: frozenset[str],
      inferred_decision_texts: frozenset[str],
      allowed_evidence_ref_ids: frozenset[str],
      existing_constraint_texts: set[str],
  ) -> None:
    for constraint in constraints:
      normalized_text = self._normalize_text(constraint.text)
      if normalized_text in existing_constraint_texts:
        continue
      if normalized_text in inferred_decision_texts:
        raise ValueError(
            'Inferred statements must be stored in hypotheses, not constraints.'
        )
      if constraint.evidence_refs:
        for evidence_ref in constraint.evidence_refs:
          if evidence_ref.ref_id not in allowed_evidence_ref_ids:
            raise ValueError(
                'Constraint evidence ref id is not in allowed evidence set.'
            )
        continue
      if normalized_text not in explicit_instruction_texts:
        raise ValueError(
            'Constraints must include evidence_refs or explicit user '
            'instruction.'
        )

  def _apply_token_budget(self, task_state: TaskStateAnchor) -> TaskStateAnchor:
    capped_state = task_state.model_copy(deep=True)
    while self._estimate_tokens(capped_state) > self._task_state_token_budget:
      if capped_state.hypotheses:
        capped_state.hypotheses.pop(0)
        continue
      if capped_state.current_plan:
        capped_state.current_plan.pop(0)
        continue
      if capped_state.next_steps:
        capped_state.next_steps.pop(0)
        continue
      if capped_state.known_failures:
        capped_state.known_failures.pop(0)
        continue
      if capped_state.constraints:
        capped_state.constraints.pop(0)
        continue
      objective_words = capped_state.objective.split()
      if len(objective_words) <= 3:
        break
      capped_state.objective = ' '.join(objective_words[:-1])
    return capped_state

  def _estimate_tokens(self, task_state: TaskStateAnchor) -> int:
    return max(1, len(task_state.model_dump_json(by_alias=True)) // 4)

  def _observation_evidence_ref_ids(
      self, observation: Observation
  ) -> frozenset[str]:
    ref_ids = {
        evidence_ref.ref_id for evidence_ref in observation.evidence_refs
    }
    ref_ids.add(observation.observation_id)
    for decision in observation.decisions + observation.learned_constraints:
      ref_ids.update(
          evidence_ref.ref_id for evidence_ref in decision.evidence_refs
      )
    for item in observation.open_questions + observation.next_steps:
      ref_ids.update(evidence_ref.ref_id for evidence_ref in item.evidence_refs)
    return frozenset(ref_ids)

  def _inferred_decision_texts(
      self, observation: Observation
  ) -> frozenset[str]:
    inferred_texts = {
        self._normalize_text(decision.text)
        for decision in observation.decisions + observation.learned_constraints
        if decision.kind == 'inferred'
    }
    return frozenset(text for text in inferred_texts if text)

  def _merge_items(
      self,
      existing: Sequence[EvidencedItem],
      additions: Sequence[EvidencedItem],
  ) -> list[EvidencedItem]:
    merged_by_text: dict[str, EvidencedItem] = {}
    ordered_keys: list[str] = []
    for item in list(existing) + list(additions):
      normalized_text = self._normalize_text(item.text)
      if not normalized_text:
        continue
      if normalized_text in merged_by_text:
        prior_item = merged_by_text[normalized_text]
        merged_by_text[normalized_text] = self._merge_evidence_refs(
            prior_item=prior_item,
            new_item=item,
        )
        continue
      ordered_keys.append(normalized_text)
      merged_by_text[normalized_text] = item.model_copy(deep=True)

    return [merged_by_text[key] for key in ordered_keys]

  def _merge_evidence_refs(
      self,
      *,
      prior_item: EvidencedItem,
      new_item: EvidencedItem,
  ) -> EvidencedItem:
    refs: list[EvidenceRef] = []
    seen_refs: set[tuple[str, str]] = set()
    for evidence_ref in prior_item.evidence_refs + new_item.evidence_refs:
      ref_key = (evidence_ref.ref_type, evidence_ref.ref_id)
      if ref_key in seen_refs:
        continue
      seen_refs.add(ref_key)
      refs.append(evidence_ref)
    return EvidencedItem(text=prior_item.text, evidence_refs=refs)

  def _normalized_texts(self, texts: Sequence[str]) -> frozenset[str]:
    return frozenset(
        self._normalize_text(text)
        for text in texts
        if self._normalize_text(text)
    )

  def _normalize_text(self, text: str) -> str:
    return ' '.join(text.strip().lower().split())
