#!/bin/bash
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

set -u -o pipefail

DEFAULT_ITERATIONS=30
DEFAULT_PYTEST_BIN="./.venv/bin/pytest"
DEFAULT_TEST_TARGET="tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration"

iterations="$DEFAULT_ITERATIONS"
pytest_bin="$DEFAULT_PYTEST_BIN"
test_target="$DEFAULT_TEST_TARGET"
stop_on_failure=false

print_usage() {
  cat <<EOF
Usage: $0 [--iterations N] [--pytest-bin PATH] [--test-target TARGET] [--stop-on-failure]

Runs repeated hybrid compaction canary tests and prints a deterministic summary.

Options:
  --iterations N       Number of loop iterations (default: ${DEFAULT_ITERATIONS})
  --pytest-bin PATH    Pytest binary path (default: ${DEFAULT_PYTEST_BIN})
  --test-target TARGET Pytest test target (default: ${DEFAULT_TEST_TARGET})
  --stop-on-failure    Stop the loop at first failing run
  --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --iterations." >&2
        print_usage
        exit 2
      fi
      iterations="$2"
      shift 2
      ;;
    --pytest-bin)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --pytest-bin." >&2
        print_usage
        exit 2
      fi
      pytest_bin="$2"
      shift 2
      ;;
    --test-target)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --test-target." >&2
        print_usage
        exit 2
      fi
      test_target="$2"
      shift 2
      ;;
    --stop-on-failure)
      stop_on_failure=true
      shift
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage
      exit 2
      ;;
  esac
done

if ! [[ "$iterations" =~ ^[1-9][0-9]*$ ]]; then
  echo "--iterations must be a positive integer. Got: $iterations" >&2
  exit 2
fi

pass_count=0
fail_count=0
run_count=0

echo "Starting hybrid canary soak"
echo "iterations=${iterations} pytest_bin=${pytest_bin} test_target=${test_target}"

for ((i = 1; i <= iterations; i++)); do
  echo "[${i}/${iterations}] running canary test"
  if "$pytest_bin" "$test_target" -q; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
    if [[ "$stop_on_failure" == true ]]; then
      run_count=$i
      break
    fi
  fi
  run_count=$i
done

echo ""
echo "CANARY_SOAK_SUMMARY run_count=${run_count} pass_count=${pass_count} fail_count=${fail_count}"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi

exit 0
