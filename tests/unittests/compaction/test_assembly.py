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

from google.adk.compaction.assembly import HybridPromptAssembler
import inspect
from google.adk.compaction.models import CompactionStats
from google.adk.compaction.models import Decision
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import FileLineRef
from google.adk.compaction.models import Observation
from google.adk.compaction.models import PatchCompaction
from google.adk.compaction.models import Provenance
from google.adk.compaction.models import Reflection
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.models import ToolRunCompaction
from google.adk.compaction.storage.in_memory_compaction_service import InMemoryCompactionService
from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents as contents_module
from google.genai import types
import pytest


def _stats(compact_tokens_est: int = 20) -> CompactionStats:
  return CompactionStats(
      raw_tokens_est=100,
      compact_tokens_est=compact_tokens_est,
      compression_ratio=5.0,
  )


def _content_tokens(prompt_contents: list[types.Content]) -> int:
  text = ''
  for content in prompt_contents:
    for part in content.parts:
      if part.text:
        text += part.text
  if not text:
    return 0
  return max(1, len(text) // 4)


def _make_turn(*, i: int, author: str) -> Event:
  text = (
      f'{author} turn {i}: '
      'this is a long debugging message for token accounting.'
  )
  return Event(
      invocation_id=f'inv-{i}',
      author=author,
      content=types.Content(
          role='user' if author == 'user' else 'model',
          parts=[types.Part(text=text)],
      ),
  )


@pytest.mark.asyncio
async def test_hybrid_prompt_assembly_orders_layers_and_inserts_evidence_pack():
  service = InMemoryCompactionService()
  function_response_event = Event(
      invocation_id='inv-tool',
      author='agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='call-1',
                      name='run_shell',
                      response={'stderr': 'AssertionError: failed'},
                  )
              )
          ],
      ),
  )
  events = [
      _make_turn(i=1, author='user'),
      _make_turn(i=2, author='agent'),
      function_response_event,
      _make_turn(i=4, author='user'),
  ]

  tool_compaction = ToolRunCompaction(
      event_id=function_response_event.id,
      compaction_version=1,
      command='pytest tests/test_a.py::test_b',
      exit_code=1,
      error_signatures=['AssertionError'],
      key_errors=['assert 1 == 2'],
      tests_failed=['tests/test_a.py::test_b'],
      file_line_refs=[FileLineRef(path='src/demo.py', line=18)],
      trimmed_trace=['Traceback ...'],
      salient_snippets=['assert 1 == 2'],
      stats=_stats(),
      provenance=Provenance(event_id=function_response_event.id),
  )
  patch_compaction = PatchCompaction(
      event_id=function_response_event.id,
      compaction_version=1,
      files_changed=['src/demo.py'],
      hunks=[],
      semantic_tags=['bugfix'],
      stats=_stats(),
      provenance=Provenance(event_id=function_response_event.id),
  )
  await service.save_tool_run_compaction(tool_compaction)
  await service.save_patch_compaction(patch_compaction)

  observation = Observation(
      observation_id='obs-1',
      session_id='session-1',
      start_seq=1,
      end_seq=10,
      text='Observed repeated AssertionError in src/demo.py.',
      decisions=[
          Decision(
              text='Retry with focused test first.',
              kind='explicit',
              evidence_refs=[
                  EvidenceRef(
                      ref_type='event', ref_id=function_response_event.id
                  )
              ],
          )
      ],
      learned_constraints=[],
      open_questions=[],
      next_steps=[],
      evidence_refs=[
          EvidenceRef(ref_type='event', ref_id=function_response_event.id)
      ],
  )
  reflection = Reflection(
      reflection_id='refl-1',
      session_id='session-1',
      covers_observation_ids=['obs-1'],
      text='Repeated failures suggest narrow-scope verification first.',
      stable_facts=[],
      recurring_failures=[],
      strategy_updates=[],
      evidence_refs=[EvidenceRef(ref_type='observation', ref_id='obs-1')],
  )
  task_state = TaskStateAnchor(
      session_id='session-1',
      objective='Fix failing test with reliable evidence.',
      constraints=[
          EvidencedItem(
              text=(
                  'Investigate AssertionError in tests/test_a.py::test_b at '
                  'src/demo.py.'
              ),
              evidence_refs=[],
          )
      ],
      hypotheses=[],
      known_failures=[],
      current_plan=[],
      next_steps=[],
      last_updated_seq=11,
  )
  await service.save_observation(observation)
  await service.save_reflection(reflection)
  await service.save_task_state(task_state)

  baseline_contents = contents_module._get_contents(None, events, 'agent')
  assembler = HybridPromptAssembler(
      compaction_service=service,
      prompt_token_budget=1_500,
      raw_turns_count=3,
      compactions_count=3,
      observations_count=3,
      rehydration_evidence_token_budget=500,
  )

  assembled_contents = await assembler.assemble(
      session_id='session-1',
      events=events,
      baseline_contents=baseline_contents,
      system_instruction='Follow coding best practices.',
  )

  prompt_texts = [
      part.text
      for content in assembled_contents
      for part in content.parts
      if part.text
  ]
  task_state_idx = next(
      i for i, text in enumerate(prompt_texts) if text.startswith('[TaskState]')
  )
  reflection_idx = next(
      i
      for i, text in enumerate(prompt_texts)
      if text.startswith('[Reflection]')
  )
  observations_idx = next(
      i
      for i, text in enumerate(prompt_texts)
      if text.startswith('[Observations]')
  )
  evidence_idx = next(
      i
      for i, text in enumerate(prompt_texts)
      if text.startswith('[Evidence Pack]')
  )
  compactions_idx = next(
      i
      for i, text in enumerate(prompt_texts)
      if text.startswith('[Recent Compactions]')
  )

  assert task_state_idx < reflection_idx < observations_idx < evidence_idx
  assert compactions_idx > evidence_idx


@pytest.mark.asyncio
async def test_hybrid_prompt_assembly_reduces_prompt_tokens_over_20_turn_session():
  service = InMemoryCompactionService()
  events: list[Event] = []

  for i in range(1, 25):
    events.append(_make_turn(i=i * 2 - 1, author='user'))
    events.append(_make_turn(i=i * 2, author='agent'))

  events.append(
      Event(
          invocation_id='inv-tool',
          author='agent',
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='call-2',
                          name='run_shell',
                          args={
                              'command': 'pytest tests/test_long.py::test_loop'
                          },
                      )
                  )
              ],
          ),
      )
  )

  tool_event = Event(
      invocation_id='inv-tool',
      author='agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='call-2',
                      name='run_shell',
                      response={'stderr': 'AssertionError: still failing'},
                  )
              )
          ],
      ),
  )
  events.append(tool_event)

  await service.save_tool_run_compaction(
      ToolRunCompaction(
          event_id=tool_event.id,
          compaction_version=1,
          command='pytest tests/test_long.py::test_loop',
          exit_code=1,
          error_signatures=['AssertionError'],
          key_errors=['assert something failed'],
          tests_failed=['tests/test_long.py::test_loop'],
          file_line_refs=[FileLineRef(path='src/loop.py', line=40)],
          trimmed_trace=['Traceback ...'],
          salient_snippets=['assert failure'],
          stats=_stats(18),
          provenance=Provenance(event_id=tool_event.id),
      )
  )

  await service.save_observation(
      Observation(
          observation_id='obs-long-1',
          session_id='session-2',
          start_seq=10,
          end_seq=40,
          text='Recent loop repeats AssertionError on src/loop.py.',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[EvidenceRef(ref_type='event', ref_id=tool_event.id)],
      )
  )
  await service.save_task_state(
      TaskStateAnchor(
          session_id='session-2',
          objective='Resolve the loop failure quickly.',
          constraints=[],
          hypotheses=[],
          known_failures=[],
          current_plan=[
              EvidencedItem(
                  text='Run focused pytest and inspect loop.py line 40.',
                  evidence_refs=[
                      EvidenceRef(ref_type='event', ref_id=tool_event.id)
                  ],
              )
          ],
          next_steps=[],
          last_updated_seq=41,
      )
  )

  baseline_contents = contents_module._get_contents(None, events, 'agent')
  baseline_tokens = _content_tokens(baseline_contents)

  assembler = HybridPromptAssembler(
      compaction_service=service,
      prompt_token_budget=700,
      raw_turns_count=4,
      compactions_count=4,
      observations_count=3,
      rehydration_evidence_token_budget=300,
  )
  hybrid_contents = await assembler.assemble(
      session_id='session-2',
      events=events,
      baseline_contents=baseline_contents,
      system_instruction='You are a coding assistant.',
  )
  hybrid_tokens = _content_tokens(hybrid_contents)

  assert len(events) >= 20
  assert hybrid_tokens < baseline_tokens


def test_hybrid_prompt_assembly_updates_remaining_tokens_for_compactions_layer():
  assemble_source = inspect.getsource(HybridPromptAssembler.assemble)
  assert "remaining_tokens = _append_text_layer(" in assemble_source
  assert "title='Recent Compactions'" in assemble_source
