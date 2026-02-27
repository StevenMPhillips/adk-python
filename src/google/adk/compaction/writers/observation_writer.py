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
from dataclasses import dataclass
import json
import re

from google.genai import types

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ..models import Decision
from ..models import EvidencedItem
from ..models import EvidenceRef
from ..models import Observation
from ..models import PatchCompaction
from ..models import TaskStateAnchor
from ..models import ToolRunCompaction
from ..storage.base_compaction_service import BaseCompactionService


@dataclass(frozen=True)
class RawTurn:
  """Minimal raw-turn shape used by observational memory compaction."""

  session_id: str
  seq: int
  event_id: str
  author: str
  text: str
  raw_tokens_est: int


class ObservationWriter:
  """Generates and persists observations from recent session activity."""

  _DEFAULT_RAW_TOKEN_THRESHOLD = 1_200

  def __init__(
      self,
      *,
      llm: BaseLlm,
      compaction_service: BaseCompactionService,
      raw_token_threshold: int = _DEFAULT_RAW_TOKEN_THRESHOLD,
  ):
    if raw_token_threshold <= 0:
      raise ValueError('raw_token_threshold must be greater than zero.')
    self._llm = llm
    self._compaction_service = compaction_service
    self._raw_token_threshold = raw_token_threshold

  def should_write_observation(
      self,
      *,
      recent_raw_turns: Sequence[RawTurn],
      episode_closed: bool,
  ) -> bool:
    """Returns whether observation generation should run for this window."""
    if not recent_raw_turns:
      return False
    raw_tokens_est = sum(turn.raw_tokens_est for turn in recent_raw_turns)
    return raw_tokens_est >= self._raw_token_threshold or episode_closed

  async def maybe_write_observation(
      self,
      *,
      recent_raw_turns: Sequence[RawTurn],
      recent_tool_run_compactions: Sequence[ToolRunCompaction],
      recent_patch_compactions: Sequence[PatchCompaction],
      current_task_state: TaskStateAnchor,
      last_observation: Observation | None = None,
      episode_closed: bool = False,
  ) -> Observation | None:
    """Writes one observation when trigger conditions are met."""
    if not self.should_write_observation(
        recent_raw_turns=recent_raw_turns,
        episode_closed=episode_closed,
    ):
      return None

    self._validate_inputs(
        recent_raw_turns=recent_raw_turns,
        current_task_state=current_task_state,
    )
    start_seq = min(turn.seq for turn in recent_raw_turns)
    end_seq = max(turn.seq for turn in recent_raw_turns)
    evidence_ref_ids = self._allowed_evidence_ref_ids(
        recent_raw_turns=recent_raw_turns,
        recent_tool_run_compactions=recent_tool_run_compactions,
        recent_patch_compactions=recent_patch_compactions,
        current_task_state=current_task_state,
        last_observation=last_observation,
    )

    prompt = self._build_prompt(
        recent_raw_turns=recent_raw_turns,
        recent_tool_run_compactions=recent_tool_run_compactions,
        recent_patch_compactions=recent_patch_compactions,
        current_task_state=current_task_state,
        last_observation=last_observation,
        start_seq=start_seq,
        end_seq=end_seq,
        evidence_ref_ids=evidence_ref_ids,
    )
    llm_request = LlmRequest(
        model=self._llm.model,
        contents=[types.Content(role='user', parts=[types.Part(text=prompt)])],
    )
    llm_request.set_output_schema(Observation)
    raw_json = await self._generate_json_output(llm_request)
    generated_observation = Observation.model_validate_json(raw_json)
    generated_observation = self._sanitize_observation(
        observation=generated_observation,
        evidence_ref_ids=evidence_ref_ids,
    )
    if self._should_skip_observation(generated_observation):
      return None
    self._validate_observation(
        observation=generated_observation,
        evidence_ref_ids=evidence_ref_ids,
    )

    observation_to_save = generated_observation.model_copy(
        update={
            'session_id': current_task_state.session_id,
            'start_seq': start_seq,
            'end_seq': end_seq,
        }
    )
    await self._compaction_service.save_observation(observation_to_save)
    return observation_to_save

  def _sanitize_observation(
      self,
      *,
      observation: Observation,
      evidence_ref_ids: frozenset[str],
  ) -> Observation:
    sanitized_decisions = self._sanitize_decisions(
        observation.decisions,
        evidence_ref_ids=evidence_ref_ids,
    )
    sanitized_constraints = self._sanitize_decisions(
        observation.learned_constraints,
        evidence_ref_ids=evidence_ref_ids,
    )
    sanitized_questions = self._sanitize_items(
        observation.open_questions,
        evidence_ref_ids=evidence_ref_ids,
    )
    sanitized_next_steps = self._sanitize_items(
        observation.next_steps,
        evidence_ref_ids=evidence_ref_ids,
    )
    sanitized_top_refs = self._sanitize_evidence_refs(
        observation.evidence_refs,
        evidence_ref_ids=evidence_ref_ids,
    )

    return observation.model_copy(
        update={
            'decisions': sanitized_decisions,
            'learned_constraints': sanitized_constraints,
            'open_questions': sanitized_questions,
            'next_steps': sanitized_next_steps,
            'evidence_refs': sanitized_top_refs,
        }
    )

  def _should_skip_observation(self, observation: Observation) -> bool:
    if observation.decisions or observation.learned_constraints:
      return False
    for item in observation.open_questions + observation.next_steps:
      if item.evidence_refs:
        return False
    return not observation.evidence_refs

  def _sanitize_decisions(
      self,
      decisions: Sequence[Decision],
      *,
      evidence_ref_ids: frozenset[str],
  ) -> list[Decision]:
    sanitized_decisions: list[Decision] = []
    for decision in decisions:
      sanitized_refs = self._sanitize_evidence_refs(
          decision.evidence_refs,
          evidence_ref_ids=evidence_ref_ids,
      )
      if not sanitized_refs:
        continue
      sanitized_decisions.append(
          decision.model_copy(update={'evidence_refs': sanitized_refs})
      )
    return sanitized_decisions

  def _sanitize_items(
      self,
      items: Sequence[EvidencedItem],
      *,
      evidence_ref_ids: frozenset[str],
  ) -> list[EvidencedItem]:
    sanitized_items: list[EvidencedItem] = []
    for item in items:
      sanitized_refs = self._sanitize_evidence_refs(
          item.evidence_refs,
          evidence_ref_ids=evidence_ref_ids,
      )
      sanitized_items.append(item.model_copy(update={'evidence_refs': sanitized_refs}))
    return sanitized_items

  def _sanitize_evidence_refs(
      self,
      evidence_refs: Sequence[EvidenceRef],
      *,
      evidence_ref_ids: frozenset[str],
  ) -> list[EvidenceRef]:
    normalized_allowed = {
        ref_id.casefold(): ref_id for ref_id in evidence_ref_ids
    }
    sanitized_refs: list[EvidenceRef] = []
    seen_refs: set[tuple[str, str]] = set()
    for evidence_ref in evidence_refs:
      canonical_ref_id = self._canonicalize_ref_id(
          evidence_ref.ref_id,
          evidence_ref_ids=evidence_ref_ids,
          normalized_allowed=normalized_allowed,
      )
      if canonical_ref_id is None:
        continue
      dedupe_key = (evidence_ref.ref_type, canonical_ref_id)
      if dedupe_key in seen_refs:
        continue
      seen_refs.add(dedupe_key)
      sanitized_refs.append(
          evidence_ref.model_copy(update={'ref_id': canonical_ref_id})
      )
    return sanitized_refs

  def _canonicalize_ref_id(
      self,
      ref_id: str,
      *,
      evidence_ref_ids: frozenset[str],
      normalized_allowed: dict[str, str],
  ) -> str | None:
    normalized_ref_id = ref_id.strip().strip('`"\'')
    if not normalized_ref_id:
      return None
    if normalized_ref_id in evidence_ref_ids:
      return normalized_ref_id

    casefold_id = normalized_ref_id.casefold()
    if casefold_id in normalized_allowed:
      return normalized_allowed[casefold_id]

    for token in re.split(r'[^A-Za-z0-9:_\-]+', normalized_ref_id):
      if not token:
        continue
      if token in evidence_ref_ids:
        return token
      token_casefold = token.casefold()
      if token_casefold in normalized_allowed:
        return normalized_allowed[token_casefold]

    matching_ids: list[str] = []
    for allowed_ref_id in evidence_ref_ids:
      match = re.search(
          rf'(^|[^A-Za-z0-9_\-]){re.escape(allowed_ref_id)}'
          r'([^A-Za-z0-9_\-]|$)',
          normalized_ref_id,
      )
      if match:
        matching_ids.append(allowed_ref_id)
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
    raise ValueError(
        'ObservationWriter LLM response did not include JSON text.'
    )

  def _validate_inputs(
      self,
      *,
      recent_raw_turns: Sequence[RawTurn],
      current_task_state: TaskStateAnchor,
  ) -> None:
    session_ids = {turn.session_id for turn in recent_raw_turns}
    if len(session_ids) != 1:
      raise ValueError('All recent_raw_turns must belong to one session.')
    session_id = next(iter(session_ids))
    if session_id != current_task_state.session_id:
      raise ValueError('Task state session does not match recent_raw_turns.')

  def _allowed_evidence_ref_ids(
      self,
      *,
      recent_raw_turns: Sequence[RawTurn],
      recent_tool_run_compactions: Sequence[ToolRunCompaction],
      recent_patch_compactions: Sequence[PatchCompaction],
      current_task_state: TaskStateAnchor,
      last_observation: Observation | None,
  ) -> frozenset[str]:
    evidence_ref_ids = {turn.event_id for turn in recent_raw_turns}
    evidence_ref_ids.update(
        compaction.event_id for compaction in recent_tool_run_compactions
    )
    evidence_ref_ids.update(
        compaction.event_id for compaction in recent_patch_compactions
    )
    evidence_ref_ids.add(f'task-state:{current_task_state.last_updated_seq}')
    if last_observation is not None:
      evidence_ref_ids.add(last_observation.observation_id)
    return frozenset(evidence_ref_ids)

  def _build_prompt(
      self,
      *,
      recent_raw_turns: Sequence[RawTurn],
      recent_tool_run_compactions: Sequence[ToolRunCompaction],
      recent_patch_compactions: Sequence[PatchCompaction],
      current_task_state: TaskStateAnchor,
      last_observation: Observation | None,
      start_seq: int,
      end_seq: int,
      evidence_ref_ids: frozenset[str],
  ) -> str:
    raw_turn_lines = [
        {
            'seq': turn.seq,
            'eventId': turn.event_id,
            'author': turn.author,
            'text': turn.text,
            'rawTokensEst': turn.raw_tokens_est,
        }
        for turn in recent_raw_turns
    ]
    tool_run_lines = [
        compaction.model_dump(mode='json', by_alias=True)
        for compaction in recent_tool_run_compactions
    ]
    patch_lines = [
        compaction.model_dump(mode='json', by_alias=True)
        for compaction in recent_patch_compactions
    ]
    payload = {
        'sessionId': current_task_state.session_id,
        'targetSeqRange': {'startSeq': start_seq, 'endSeq': end_seq},
        'recentRawTurns': raw_turn_lines,
        'recentToolRunCompactions': tool_run_lines,
        'recentPatchCompactions': patch_lines,
        'currentTaskState': current_task_state.model_dump(
            mode='json', by_alias=True
        ),
        'lastObservation': (
            None
            if last_observation is None
            else last_observation.model_dump(mode='json', by_alias=True)
        ),
        'allowedEvidenceRefIds': sorted(evidence_ref_ids),
    }
    context_json = json.dumps(payload, indent=2, sort_keys=True)
    return (
        'Generate exactly one Observation JSON object for the target seq'
        ' range. Use only evidence present in the provided context. Every item'
        ' in decisions and learnedConstraints must include at least one'
        ' evidenceRefs entry. Set decision.kind to "explicit" only when the'
        ' decision is directly evidenced by cited material; otherwise set'
        ' decision.kind to "inferred". Every evidenceRefs.refId must be in'
        f' allowedEvidenceRefIds. Return JSON only.\n\nContext:\n{context_json}'
    )

  def _validate_observation(
      self,
      *,
      observation: Observation,
      evidence_ref_ids: frozenset[str],
  ) -> None:
    for decision in observation.decisions + observation.learned_constraints:
      if not decision.evidence_refs:
        raise ValueError('Each decision/constraint must include evidence_refs.')
      for evidence_ref in decision.evidence_refs:
        if evidence_ref.ref_id not in evidence_ref_ids:
          raise ValueError(
              'Decision evidence ref id is not in allowed evidence set.'
          )

    for item in observation.open_questions + observation.next_steps:
      for evidence_ref in item.evidence_refs:
        if evidence_ref.ref_id not in evidence_ref_ids:
          raise ValueError(
              'Observation item evidence ref id is not in allowed evidence set.'
          )

    for evidence_ref in observation.evidence_refs:
      if evidence_ref.ref_id not in evidence_ref_ids:
        raise ValueError(
            'Top-level evidence ref id is not in allowed evidence set.'
        )
