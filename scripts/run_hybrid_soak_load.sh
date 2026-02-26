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

DEFAULT_ITERATIONS=100
DEFAULT_WORKERS=1
DEFAULT_PYTEST_BIN="./.venv/bin/pytest"
DEFAULT_TEST_TARGET="tests/unittests/runners/test_hybrid_compaction.py::test_hybrid_compaction_end_to_end_observational_integration"

iterations="$DEFAULT_ITERATIONS"
workers="$DEFAULT_WORKERS"
pytest_bin="$DEFAULT_PYTEST_BIN"
test_target="$DEFAULT_TEST_TARGET"
stop_on_failure=false

print_usage() {
  cat <<EOF
Usage: $0 [--iterations N] [--workers N] [--pytest-bin PATH] [--test-target TARGET] [--stop-on-failure]

Runs deterministic hybrid-session soak/load loops against an existing hybrid test target.

Options:
  --iterations N       Number of total runs (default: ${DEFAULT_ITERATIONS})
  --workers N          Number of concurrent workers (default: ${DEFAULT_WORKERS})
  --pytest-bin PATH    Pytest binary path (default: ${DEFAULT_PYTEST_BIN})
  --test-target TARGET Pytest target (default: ${DEFAULT_TEST_TARGET})
  --stop-on-failure    Stop scheduling new runs after first observed failure
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
    --workers)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --workers." >&2
        print_usage
        exit 2
      fi
      workers="$2"
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

if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer. Got: $workers" >&2
  exit 2
fi

pass_count=0
fail_count=0
run_count=0
next_iteration=1

log_files=()
SECONDS=0

cleanup() {
  for log_file in "${log_files[@]}"; do
    if [[ -f "$log_file" ]]; then
      rm -f "$log_file"
    fi
  done
}

trap cleanup EXIT

echo "Starting hybrid soak/load"
echo "iterations=${iterations} workers=${workers} pytest_bin=${pytest_bin} test_target=${test_target}"

should_stop=false

while [[ "$next_iteration" -le "$iterations" ]]; do
  batch_pids=()
  batch_indices=()
  batch_logs=()
  launched=0

  while [[ "$launched" -lt "$workers" && "$next_iteration" -le "$iterations" ]]; do
    run_index="$next_iteration"
    next_iteration=$((next_iteration + 1))
    launched=$((launched + 1))

    log_file="$(mktemp -t hybrid_soak_load.XXXXXX)"
    log_files+=("$log_file")

    echo "[${run_index}/${iterations}] started"
    (
      "$pytest_bin" "$test_target" -q
    ) >"$log_file" 2>&1 &

    batch_pids+=("$!")
    batch_indices+=("$run_index")
    batch_logs+=("$log_file")
  done

  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    run_index="${batch_indices[$i]}"
    log_file="${batch_logs[$i]}"

    if wait "$pid"; then
      pass_count=$((pass_count + 1))
      echo "[${run_index}/${iterations}] pass"
    else
      fail_count=$((fail_count + 1))
      echo "[${run_index}/${iterations}] fail"
      echo "----- failure output (run ${run_index}) -----"
      cat "$log_file"
      echo "----- end failure output (run ${run_index}) -----"
      if [[ "$stop_on_failure" == true ]]; then
        should_stop=true
      fi
    fi

    run_count=$((run_count + 1))
  done

  if [[ "$should_stop" == true ]]; then
    break
  fi
done

duration_seconds="$SECONDS"

echo ""
echo "HYBRID_SOAK_LOAD_SUMMARY run_count=${run_count} pass_count=${pass_count} fail_count=${fail_count} duration_seconds=${duration_seconds}"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi

exit 0
