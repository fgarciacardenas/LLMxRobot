#!/usr/bin/env bash
set -euo pipefail

# Always run relative to this script's directory so imports/paths work
# regardless of the caller's current working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
elif [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
else
  echo "⚠️ .env file not found. Make sure OPENAI_API_KEY is set in the environment."
fi

# Define a base directory for logs
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

timestamp() { date +'%Y%m%d_%H%M%S'; }

# Base command
BASE_CMD='python3 -m tests.decision_tester.decision_tester \
  --dataset all \
  --ax_local \
  --local_workdir ~/RISCVxLLMxRobot/third_party/voyager-sdk \
  --local_venv ~/RISCVxLLMxRobot/third_party/voyager-sdk/venv/bin/activate \
  --local_run "export OPENAI_API_KEY=$OPENAI_API_KEY && python3 ~/RISCVxLLMxRobot/third_party/voyager-sdk/inference_llm.py phi3-mini-2048-4core-static"'

run_test() {
  local name="$1"
  local args="$2"
  local log="$3"

  echo "▶️  Running $name... Output: $log"
  # Run the command in a subshell so we can safely pipe to tee, capturing both stdout & stderr.
  # PIPESTATUS[0] gives us the exit code of the left side of the pipe (our command), not tee.
  bash -c "$BASE_CMD $args" 2>&1 | tee "$log"
  local exit_code=${PIPESTATUS[0]}

  if [[ $exit_code -ne 0 ]]; then
    echo "❌ $name failed with exit code $exit_code. See $log"
    # Append a footer marker into the log, too
    {
      echo
      echo "-----"
      echo "[$(date -Is)] $name FAILED (exit $exit_code)"
    } >> "$log"
    exit $exit_code
  else
    {
      echo
      echo "-----"
      echo "[$(date -Is)] $name PASSED"
    } >> "$log"
    echo "✅ $name completed."
  fi
}

# Test 1
LOG1="$LOG_DIR/test_rag_$(timestamp).txt"
run_test "Test 1 (--rag)" "--rag" "$LOG1"

# Test 2
LOG2="$LOG_DIR/test_rag_offline_$(timestamp).txt"
run_test "Test 2 (--rag --rag_offline)" "--rag --rag_offline" "$LOG2"

# Test 3
LOG3="$LOG_DIR/test_no_rag_$(timestamp).txt"
run_test "Test 3 (no --rag)" "" "$LOG3"

echo "🎉 All tests completed successfully."
