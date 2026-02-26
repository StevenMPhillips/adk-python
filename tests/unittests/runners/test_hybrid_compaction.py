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

import json

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.compaction.compactors.patch_compactor import PatchCompactor
from google.adk.compaction.compactors.registry import ToolRunCompactorRegistry
from google.adk.compaction.config import HybridEventsCompactionConfig
from google.adk.compaction.models import Decision
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import Observation
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.rehydration import RehydrationExecutor
from google.adk.compaction.rehydration import RetrievalQuery
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)
from google.adk.compaction.writers.observation_writer import ObservationWriter
from google.adk.compaction.writers.observation_writer import RawTurn
from google.adk.compaction.writers.task_state_updater import TaskStateUpdater
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from google.genai.types import FunctionCall
from google.genai.types import Part
import pytest

from .. import testing_utils


def _run_shell(command: str) -> dict[str, object]:
  if command.startswith('pytest'):
    return {
        'command': command,
        'exit_code': 1,
        'stderr': (
            'tests/unittests/runners/test_hybrid_compaction.py:99: '
            'AssertionError'
        ),
        'stdout': '',
        'duration_ms': 27,
    }

  patch = '\n'.join([
      'diff --git a/src/demo.py b/src/demo.py',
      '--- a/src/demo.py',
      '+++ b/src/demo.py',
      '@@ -1,1 +1,1 @@',
      '-print("old")',
      '+print("new")',
  ])
  return {
      'command': command,
      'exit_code': 0,
      'diff': patch,
      'stdout': patch,
  }


def _debug_shell(command: str) -> dict[str, object]:
  if command.startswith('pytest'):
    attempt = 0
    if '--attempt=' in command:
      attempt = int(command.rsplit('--attempt=', 1)[1])
    failure_line = 40 + attempt
    stdout = '\n'.join([
        '============================= FAILURES ==============================',
        (
            'FAILED tests/integration/test_debug_flow.py::test_retry_loop - '
            'AssertionError: expected compacted evidence'
        ),
        f'src/app/core.py:{failure_line}: AssertionError',
    ])
    stderr = '\n'.join([
        'Traceback (most recent call last):',
        f'  File "src/app/core.py", line {failure_line}, in test_retry_loop',
        '    assert compacted_refs == raw_refs',
        'E   AssertionError: expected compacted evidence',
    ])
    return {
        'command': command,
        'exit_code': 1,
        'stderr': stderr,
        'stdout': stdout,
        'duration_ms': 20 + attempt,
    }

  patch_id = 0
  if '--patch=' in command:
    patch_id = int(command.rsplit('--patch=', 1)[1])
  before_line = f'  return _rehydrate_previous(signature, {patch_id})'
  after_line = (
      f'  return _rehydrate_previous(signature, {patch_id}, include_raw=True)'
  )
  patch = '\n'.join([
      'diff --git a/src/app/core.py b/src/app/core.py',
      '--- a/src/app/core.py',
      '+++ b/src/app/core.py',
      '@@ -18,7 +18,7 @@ def resolve_signature(signature: str) -> str:',
      f'-{before_line}',
      f'+{after_line}',
  ])
  return {
      'command': command,
      'exit_code': 0,
      'diff': patch,
      'stdout': patch,
      'duration_ms': 12,
  }


class _ObservationJsonModel(BaseLlm):
  model: str = 'observation-json'

  @classmethod
  def supported_models(cls) -> list[str]:
    return ['observation-json']

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ):
    del stream
    prompt_text = llm_request.contents[0].parts[0].text or ''
    context_json = prompt_text.split('Context:\n', 1)[1]
    context = json.loads(context_json)
    allowed_refs = context['allowedEvidenceRefIds']
    evidence_ref_id = allowed_refs[0]
    observation = Observation(
        session_id='placeholder-session',
        start_seq=0,
        end_seq=0,
        text='Observed repeated pytest failures while patching core logic.',
        decisions=[
            Decision(
                text='Keep reproducing the same failing test during iteration.',
                kind='explicit',
                evidence_refs=[
                    EvidenceRef(ref_type='event', ref_id=evidence_ref_id)
                ],
            )
        ],
        learned_constraints=[
            Decision(
                text=(
                    'Failure signatures should stay queryable after compaction.'
                ),
                kind='inferred',
                evidence_refs=[
                    EvidenceRef(ref_type='event', ref_id=evidence_ref_id)
                ],
            )
        ],
        open_questions=[
            EvidencedItem(
                text='Which earlier run first showed this signature?',
                evidence_refs=[
                    EvidenceRef(ref_type='event', ref_id=evidence_ref_id)
                ],
            )
        ],
        next_steps=[
            EvidencedItem(
                text='Rehydrate matching evidence before next patch.',
                evidence_refs=[
                    EvidenceRef(ref_type='event', ref_id=evidence_ref_id)
                ],
            )
        ],
        evidence_refs=[EvidenceRef(ref_type='event', ref_id=evidence_ref_id)],
    )
    yield LlmResponse(
        content=types.Content(
            role='model',
            parts=[Part(text=observation.model_dump_json(by_alias=True))],
        )
    )


def _build_turn_plan() -> list[tuple[str, str | None]]:
  plan: list[tuple[str, str | None]] = []
  for cycle in range(1, 7):
    plan.extend([
        (
            f'Run focused pytest and report failures (cycle {cycle}).',
            (
                'pytest tests/integration/test_debug_flow.py::test_retry_loop '
                f'--attempt={cycle}'
            ),
        ),
        (
            f'Summarize what broke and why (cycle {cycle}).',
            None,
        ),
        (
            f'Show diff for the latest patch (cycle {cycle}).',
            f'git diff -- src/app/core.py --patch={cycle}',
        ),
        (
            f'Re-run pytest after this patch (cycle {cycle}).',
            (
                'pytest tests/integration/test_debug_flow.py::test_retry_loop '
                f'--attempt={cycle + 20}'
            ),
        ),
    ])
  return plan


def _text_from_event(event) -> str:
  if not event.content or not event.content.parts:
    return ''
  return '\n'.join(part.text for part in event.content.parts if part.text)


@pytest.mark.asyncio
async def test_runner_creates_deterministic_artifacts_for_multi_turn_tool_runs(
):
  tool = FunctionTool(func=_run_shell)
  llm_responses = [
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name,
                          args={
                              'command': (
                                  'pytest tests/unittests/runners/'
                                  'test_hybrid_compaction.py'
                              )
                          },
                      )
                  )
              ]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[Part(text='turn 1 complete')]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name,
                          args={'command': 'git diff -- src/demo.py'},
                      )
                  )
              ]
          )
      ),
      testing_utils.LlmResponse(
          content=testing_utils.ModelContent(
              parts=[Part(text='turn 2 complete')]
          )
      ),
  ]
  model = testing_utils.MockModel.create(llm_responses)
  compaction_service = InMemoryCompactionService()
  app = App(
      name='hybrid_compaction_app',
      root_agent=LlmAgent(name='agent', model=model, tools=[tool]),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  await runner.run_async('run tests')
  await runner.run_async('show me the patch')

  tool_response_events = [
      event for event in runner.session.events if event.get_function_responses()
  ]
  assert len(tool_response_events) == 2

  first_tool_run = await compaction_service.get_tool_run_compaction(
      tool_response_events[0].id
  )
  second_tool_run = await compaction_service.get_tool_run_compaction(
      tool_response_events[1].id
  )
  first_patch = await compaction_service.get_patch_compaction(
      tool_response_events[0].id
  )
  second_patch = await compaction_service.get_patch_compaction(
      tool_response_events[1].id
  )

  assert first_tool_run is not None
  assert second_tool_run is not None
  assert first_tool_run.command.startswith('pytest')
  assert second_tool_run.command.startswith('git diff')
  assert first_patch is None
  assert second_patch is not None
  assert second_patch.files_changed == ['src/demo.py']


@pytest.mark.asyncio
async def test_hybrid_compaction_end_to_end_observational_integration():
  tool = FunctionTool(func=_debug_shell)
  turns = _build_turn_plan()

  llm_responses = []
  for turn_index, (_, command) in enumerate(turns, start=1):
    if command is None:
      llm_responses.append(
          testing_utils.LlmResponse(
              content=testing_utils.ModelContent(
                  parts=[Part(text=f'Debug analysis for turn {turn_index}.')]
              )
          )
      )
      continue
    llm_responses.extend([
        testing_utils.LlmResponse(
            content=testing_utils.ModelContent(
                parts=[
                    Part(
                        function_call=FunctionCall(
                            name=tool.name,
                            args={'command': command},
                        )
                    )
                ]
            )
        ),
        testing_utils.LlmResponse(
            content=testing_utils.ModelContent(
                parts=[Part(text=f'Completed tool turn {turn_index}.')]
            )
        ),
    ])

  model = testing_utils.MockModel.create(llm_responses)
  compaction_service = InMemoryCompactionService()
  app = App(
      name='hybrid_compaction_long_session_app',
      root_agent=LlmAgent(name='agent', model=model, tools=[tool]),
      events_compaction_config=HybridEventsCompactionConfig(
          compaction_service=compaction_service,
          tool_run_compactor_registry=ToolRunCompactorRegistry(),
          patch_compactor=PatchCompactor(),
          compaction_interval=999,
          overlap_size=0,
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  for user_message, _ in turns:
    await runner.run_async(user_message)

  tool_response_events = [
      event for event in runner.session.events if event.get_function_responses()
  ]
  assert len(turns) >= 20
  assert len(tool_response_events) == 18

  tool_run_compactions = []
  patch_compactions = []
  for event in tool_response_events:
    tool_run = await compaction_service.get_tool_run_compaction(event.id)
    patch = await compaction_service.get_patch_compaction(event.id)
    assert tool_run is not None
    tool_run_compactions.append(tool_run)
    if patch is not None:
      patch_compactions.append(patch)

  assert len(tool_run_compactions) == len(tool_response_events)
  assert len(patch_compactions) == 6

  raw_turns = []
  seq_by_event_id = {}
  for seq, event in enumerate(runner.session.events, start=1):
    seq_by_event_id[event.id] = seq
    text = _text_from_event(event)
    if not text:
      continue
    raw_turns.append(
        RawTurn(
            session_id=runner.session.id,
            seq=seq,
            event_id=event.id,
            author=event.author,
            text=text,
            raw_tokens_est=max(1, len(text) // 4),
        )
    )

  observation_interval = 8
  observation_writer = ObservationWriter(
      llm=_ObservationJsonModel(),
      compaction_service=compaction_service,
      raw_token_threshold=1,
  )
  task_updater = TaskStateUpdater(
      compaction_service=compaction_service,
      task_state_token_budget=5_000,
  )

  task_state = TaskStateAnchor(
      session_id=runner.session.id,
      state_version=0,
      objective='Diagnose repeated pytest failures in src/app/core.py.',
      constraints=[],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=0,
  )
  await compaction_service.save_task_state(task_state)

  for tool_run in tool_run_compactions:
    task_state = await task_updater.update_from_tool_run(
        tool_run_compaction=tool_run,
        current_task_state=task_state,
    )

  expected_observations = 0
  last_observation = None
  for start in range(0, len(raw_turns), observation_interval):
    chunk = raw_turns[start : start + observation_interval]
    if not chunk:
      continue
    expected_observations += 1
    chunk_start_seq = chunk[0].seq
    chunk_end_seq = chunk[-1].seq

    chunk_tool_runs = [
        compaction
        for compaction in tool_run_compactions
        if (
            chunk_start_seq
            <= seq_by_event_id[compaction.event_id]
            <= chunk_end_seq
        )
    ]
    chunk_patches = [
        compaction
        for compaction in patch_compactions
        if (
            chunk_start_seq
            <= seq_by_event_id[compaction.event_id]
            <= chunk_end_seq
        )
    ]

    observation = await observation_writer.maybe_write_observation(
        recent_raw_turns=chunk,
        recent_tool_run_compactions=chunk_tool_runs,
        recent_patch_compactions=chunk_patches,
        current_task_state=task_state,
        last_observation=last_observation,
        episode_closed=True,
    )
    assert observation is not None
    last_observation = observation
    task_state = await task_updater.update_from_observation(
        observation=observation,
        current_task_state=task_state,
        use_llm=False,
    )

  observations = await compaction_service.get_observations(runner.session.id)
  assert len(observations) == expected_observations

  final_task_state = await compaction_service.get_task_state(runner.session.id)
  assert final_task_state is not None
  assert final_task_state.state_version > len(tool_run_compactions)
  assert final_task_state.known_failures
  assert all(item.evidence_refs for item in final_task_state.known_failures)
  assert final_task_state.hypotheses
  assert all(item.evidence_refs for item in final_task_state.hypotheses)

  signature = (
      'pytest::AssertionError::tests/integration/test_debug_flow.py::'
      'test_retry_loop'
  )
  by_signature = await compaction_service.query_by_error_signature(signature)
  assert by_signature
  by_file = await compaction_service.query_by_file_path('src/app/core.py')
  assert by_file
  tool_match = next(
      item for item in by_signature if hasattr(item, 'file_line_refs')
  )
  assert any(ref.path == 'src/app/core.py' for ref in tool_match.file_line_refs)

  raw_excerpt_by_event_id = {}
  for event in tool_response_events:
    response = event.get_function_responses()[0].response
    if not isinstance(response, dict):
      continue
    command = response.get('command')
    if isinstance(command, str) and command.startswith('pytest'):
      raw_excerpt_by_event_id[event.id] = response.get('stderr', '')

  rehydration_executor = RehydrationExecutor(
      compaction_service=compaction_service,
      token_budget=1_200,
  )
  evidence_pack = await rehydration_executor.execute(
      [
          RetrievalQuery(
              trigger='thrash_detection',
              reason='Need the original failure payload.',
              error_signature=signature,
              file_path='src/app/core.py',
          )
      ],
      event_time_by_id=seq_by_event_id,
      raw_excerpt_by_event_id=raw_excerpt_by_event_id,
  )
  assert evidence_pack.compacted_artifacts
  assert any(
      excerpt.event_id in raw_excerpt_by_event_id
      and 'AssertionError: expected compacted evidence' in excerpt.excerpt
      for excerpt in evidence_pack.raw_excerpts
  )

  llm_only_tokens_est = sum(
      compaction.stats.raw_tokens_est for compaction in tool_run_compactions
  ) + sum(compaction.stats.raw_tokens_est for compaction in patch_compactions)
  hybrid_tokens_est = sum(
      compaction.stats.compact_tokens_est for compaction in tool_run_compactions
  ) + sum(
      compaction.stats.compact_tokens_est for compaction in patch_compactions
  )
  assert hybrid_tokens_est < llm_only_tokens_est
