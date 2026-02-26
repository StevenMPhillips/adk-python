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

from google.adk import Agent
from google.adk.apps import App
from google.adk.compaction.config import HybridEventsCompactionConfig
from google.adk.compaction.storage.in_memory_compaction_service import (
    InMemoryCompactionService,
)
from google.genai import types

_INSTRUCTION = """
You are a careful long-memory QA assistant.

Use the conversation history to answer the final user question.

Rules:
- Prefer exact facts from prior conversation turns.
- If asked for the latest version of updated information, use the most recent
  update from the timeline.
- For time-based questions, reason from the timestamps and chronology in the
  conversation history.
- If the answer is not present in the provided history, say that the question
  is not answerable from the available conversation history.
- Keep answers concise and factual.
""".strip()

root_agent = Agent(
    name='longmemeval_benchmark_agent',
    model='gemini-2.5-flash',
    description='Benchmark-oriented long memory QA agent.',
    instruction=_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

app = App(
    name='longmemeval_benchmark_app',
    root_agent=root_agent,
    events_compaction_config=HybridEventsCompactionConfig(
        compaction_service=InMemoryCompactionService(),
        compaction_interval=1_000_000,
        overlap_size=0,
    ),
)
