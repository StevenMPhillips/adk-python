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

from collections import Counter
import math

from google.adk.compaction.assembly import HybridPromptAssembler
from google.adk.compaction.models import CompactionStats
from google.adk.compaction.models import EvidenceRef
from google.adk.compaction.models import EvidencedItem
from google.adk.compaction.models import FileLineRef
from google.adk.compaction.models import Observation
from google.adk.compaction.models import PatchCompaction
from google.adk.compaction.models import Provenance
from google.adk.compaction.models import Reflection
from google.adk.compaction.models import TaskStateAnchor
from google.adk.compaction.models import ToolRunCompaction
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)
from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents as contents_module
from google.genai import types
import pytest


class _CountingCompactionService(InMemoryCompactionService):
  """Tracks deterministic read/query counts as runtime proxies."""

  def __init__(self):
    super().__init__()
    self.call_counts = Counter()

  async def get_tool_run_compaction(
      self, event_id: str
  ) -> ToolRunCompaction | None:
    self.call_counts['get_tool_run_compaction'] += 1
    return await super().get_tool_run_compaction(event_id)

  async def get_patch_compaction(self, event_id: str) -> PatchCompaction | None:
    self.call_counts['get_patch_compaction'] += 1
    return await super().get_patch_compaction(event_id)

  async def get_task_state(self, session_id: str) -> TaskStateAnchor | None:
    self.call_counts['get_task_state'] += 1
    return await super().get_task_state(session_id)

  async def get_latest_reflection(self, session_id: str) -> Reflection | None:
    self.call_counts['get_latest_reflection'] += 1
    return await super().get_latest_reflection(session_id)

  async def get_observations(
      self,
      session_id: str,
      seq_range: tuple[int, int] | None = None,
  ) -> list[Observation]:
    self.call_counts['get_observations'] += 1
    return await super().get_observations(session_id, seq_range)

  async def query_by_error_signature(
      self, signature: str
  ) -> list[ToolRunCompaction]:
    self.call_counts['query_by_error_signature'] += 1
    return await super().query_by_error_signature(signature)

  async def query_by_file_path(
      self, path: str
  ) -> list[ToolRunCompaction | PatchCompaction]:
    self.call_counts['query_by_file_path'] += 1
    return await super().query_by_file_path(path)


@pytest.fixture
def baseline_fixture() -> dict[str, int]:
  return {
      'hybrid_tokens_est': 1488,
      'tokens_reduced_permille': 459,
      'storage_reads_proxy': 18,
  }


def _content_token_proxy(prompt_contents: list[types.Content]) -> int:
  serialized = ''
  for content in prompt_contents:
    for part in content.parts:
      if part.text:
        serialized += part.text
      if part.function_call:
        serialized += part.function_call.name
        serialized += str(part.function_call.args)
      if part.function_response:
        serialized += part.function_response.name
        serialized += str(part.function_response.response)
  if not serialized:
    return 0
  return max(1, len(serialized) // 4)


def _make_text_event(*, invocation_id: str, author: str, text: str) -> Event:
  role = 'user' if author == 'user' else 'model'
  return Event(
      invocation_id=invocation_id,
      author=author,
      content=types.Content(role=role, parts=[types.Part(text=text)]),
  )


@pytest.mark.asyncio
async def test_hybrid_compaction_regression_harness_over_60_turns(
    baseline_fixture: dict[str, int],
):
  session_id = 'hybrid-perf-session'
  service = _CountingCompactionService()

  events: list[Event] = []
  tool_response_events: list[Event] = []
  total_turns = 60
  for turn in range(1, total_turns + 1):
    user_text = (
        f'user turn {turn}: inspect hybrid compaction behavior for '
        f'src/module_{turn % 4}.py.'
    )
    if turn == total_turns:
      user_text += (
          ' Please revisit older AssertionError evidence from '
          'src/module_1.py.'
      )
    events.append(
        _make_text_event(
            invocation_id=f'inv-user-{turn}',
            author='user',
            text=user_text,
        )
    )
    events.append(
        _make_text_event(
            invocation_id=f'inv-agent-{turn}',
            author='agent',
            text=(
                f'agent turn {turn}: running focused diagnostics and summarizing'
                ' findings.'
            ),
        )
    )

    if turn % 5 != 0:
      continue

    call_id = f'call-{turn}'
    events.append(
        Event(
            invocation_id=f'inv-call-{turn}',
            author='agent',
            content=types.Content(
                role='model',
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name='run_shell',
                            args={
                                'command': (
                                    'pytest '
                                    f'tests/perf_suite.py::test_case_{turn}'
                                )
                            },
                        )
                    )
                ],
            ),
        )
    )
    response_event = Event(
        invocation_id=f'inv-response-{turn}',
        author='agent',
        content=types.Content(
            role='user',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id,
                        name='run_shell',
                        response={
                            'command': (
                                'pytest '
                                f'tests/perf_suite.py::test_case_{turn}'
                            ),
                            'stderr': (
                                'AssertionError: compaction regression '
                                'candidate at src/module_1.py:42'
                            ),
                            'exit_code': 1,
                        },
                    )
                )
            ],
        ),
    )
    events.append(response_event)
    tool_response_events.append(response_event)

  assert total_turns >= 50
  assert len(tool_response_events) == 12

  for index, event in enumerate(tool_response_events, start=1):
    await service.save_tool_run_compaction(
        ToolRunCompaction(
            event_id=event.id,
            compaction_version=1,
            command=f'pytest tests/perf_suite.py::test_case_{index}',
            exit_code=1,
            error_signatures=['AssertionError'],
            key_errors=['AssertionError: compaction regression candidate'],
            tests_failed=[f'tests/perf_suite.py::test_case_{index}'],
            file_line_refs=[FileLineRef(path='src/module_1.py', line=40 + index)],
            trimmed_trace=['Traceback ... AssertionError'],
            salient_snippets=['assert baseline_tokens > hybrid_tokens'],
            stats=CompactionStats(
                raw_tokens_est=280,
                compact_tokens_est=44,
                compression_ratio=6.36,
            ),
            provenance=Provenance(event_id=event.id),
        )
    )
    if index % 2 == 0:
      await service.save_patch_compaction(
          PatchCompaction(
              event_id=event.id,
              compaction_version=1,
              files_changed=['src/module_1.py'],
              hunks=[],
              semantic_tags=['perf'],
              stats=CompactionStats(
                  raw_tokens_est=160,
                  compact_tokens_est=30,
                  compression_ratio=5.33,
              ),
              provenance=Provenance(event_id=event.id),
          )
      )

  await service.save_observation(
      Observation(
          session_id=session_id,
          start_seq=40,
          end_seq=100,
          text='AssertionError repeats while editing src/module_1.py.',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[
              EvidenceRef(ref_type='event', ref_id=tool_response_events[-1].id)
          ],
      )
  )
  await service.save_observation(
      Observation(
          session_id=session_id,
          start_seq=101,
          end_seq=170,
          text='Recent attempts keep failing the same test.',
          decisions=[],
          learned_constraints=[],
          open_questions=[],
          next_steps=[],
          evidence_refs=[
              EvidenceRef(ref_type='event', ref_id=tool_response_events[-2].id)
          ],
      )
  )
  await service.save_reflection(
      Reflection(
          session_id=session_id,
          covers_observation_ids=[],
          text='Preserve concise evidence while keeping a small raw-tail window.',
          stable_facts=[],
          recurring_failures=[],
          strategy_updates=[],
          evidence_refs=[],
      )
  )
  await service.save_task_state(
      TaskStateAnchor(
          session_id=session_id,
          state_version=12,
          objective='Prevent hybrid prompt assembly regressions.',
          constraints=[],
          hypotheses=[],
          known_failures=[],
          current_plan=[
              EvidencedItem(
                  text=(
                      'Re-check AssertionError in '
                      'tests/perf_suite.py::test_case_12 at src/module_1.py.'
                  ),
                  evidence_refs=[],
              )
          ],
          next_steps=[],
          last_updated_seq=180,
      )
  )

  baseline_contents = contents_module._get_contents(None, events, 'agent')
  assembler = HybridPromptAssembler(
      compaction_service=service,
      prompt_token_budget=1_800,
      raw_turns_count=8,
      compactions_count=8,
      observations_count=3,
      rehydration_evidence_token_budget=450,
  )
  hybrid_contents = await assembler.assemble(
      session_id=session_id,
      events=events,
      baseline_contents=baseline_contents,
      system_instruction='Keep replies concise and evidence-backed.',
  )

  baseline_tokens = _content_token_proxy(baseline_contents)
  hybrid_tokens = _content_token_proxy(hybrid_contents)
  tokens_reduced_permille = 1_000 - ((hybrid_tokens * 1_000) // baseline_tokens)
  storage_reads_proxy = sum(service.call_counts.values())

  assert len(events) >= 120
  assert baseline_tokens >= 2_000
  assert hybrid_tokens < baseline_tokens

  assert hybrid_tokens <= math.ceil(baseline_fixture['hybrid_tokens_est'] * 1.15)
  assert (
      tokens_reduced_permille
      >= math.floor(baseline_fixture['tokens_reduced_permille'] * 0.85)
  )
  assert (
      storage_reads_proxy
      <= math.ceil(baseline_fixture['storage_reads_proxy'] * 1.25)
  )
