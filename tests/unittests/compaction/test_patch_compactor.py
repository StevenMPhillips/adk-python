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

from google.genai import types
from google.adk.compaction.compactors.patch_compactor import PatchCompactor
from google.adk.events.event import Event


def _tool_event(stdout: str) -> Event:
  return Event(
      author='agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='run_shell_command',
                      response={'stdout': stdout, 'exit_code': 0},
                  )
              )
          ],
      ),
      id='evt-patch-tool',
  )


def test_patch_compactor_parses_single_file_unified_diff():
  stdout = """--- a/src/example.py
+++ b/src/example.py
@@ -1,3 +1,4 @@
 import os
-value = 1
+value = 2
+name = 'adk'
 print(value)
"""

  compaction = PatchCompactor().compact(_tool_event(stdout))

  assert compaction is not None
  assert compaction.files_changed == ['src/example.py']
  assert len(compaction.hunks) == 1
  assert compaction.hunks[0].path == 'src/example.py'
  assert compaction.hunks[0].anchor_before == 'import os'
  assert compaction.hunks[0].anchor_after == 'import os'
  assert '@@ -1,3 +1,4 @@' in compaction.hunks[0].snippet
  assert 'single-file' in compaction.semantic_tags
  assert compaction.stats.raw_tokens_est > 0
  assert compaction.stats.compact_tokens_est > 0


def test_patch_compactor_extracts_files_and_hunk_count_for_multi_file_diff():
  stdout = """--- a/src/alpha.py
+++ b/src/alpha.py
@@ -1,3 +1,3 @@
 x = 1
-y = 2
+y = 3
 z = 4
@@ -10,3 +10,4 @@
 def greet():
   return 'hello'
+
+
--- a/src/beta.py
+++ b/src/beta.py
@@ -4,2 +4,2 @@
-flag = False
+flag = True
"""

  compaction = PatchCompactor().compact(_tool_event(stdout))

  assert compaction is not None
  assert compaction.files_changed == ['src/alpha.py', 'src/beta.py']
  hunk_count_by_file = Counter(hunk.path for hunk in compaction.hunks)
  assert hunk_count_by_file['src/alpha.py'] == 2
  assert hunk_count_by_file['src/beta.py'] == 1
  assert 'multi-file' in compaction.semantic_tags


def test_patch_compactor_prioritizes_larger_hunks_when_budget_is_exceeded():
  large_hunk_lines = '\n'.join([
      '-old line 1',
      '-old line 2',
      '-old line 3',
      '-old line 4',
      '-old line 5',
      '-old line 6',
      '+new line 1',
      '+new line 2',
      '+new line 3',
      '+new line 4',
      '+new line 5',
      '+new line 6',
  ])
  stdout = """--- a/src/large.py
+++ b/src/large.py
@@ -1,6 +1,6 @@
{large_hunk}
--- a/src/small.py
+++ b/src/small.py
@@ -1,2 +1,2 @@
-a = 1
+a = 2
""".format(large_hunk=large_hunk_lines)

  compaction = PatchCompactor(token_budget=50).compact(_tool_event(stdout))

  assert compaction is not None
  assert len(compaction.hunks) == 1
  assert compaction.hunks[0].path == 'src/large.py'
  assert compaction.stats.compact_tokens_est <= 50
