#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env if it exists
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
else
  echo "⚠️ .env file not found. Make sure OPENAI_API_KEY is set in the environment."
fi

# Define a base directory for logs
LOG_DIR=./logs
mkdir -p "$LOG_DIR"

timestamp() { date +'%Y%m%d_%H%M%S'; }

# Optional: brief GPU preflight (logs to console and first test log)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "🖥️ GPU detected:"
  nvidia-smi || true
else
  echo "ℹ️ nvidia-smi not found; proceeding without GPU preflight."
fi

# Base command (OPENAI_API_KEY already in environment if .env was loaded)
BASE_CMD='python3 -m tests.decision_tester.decision_tester \
  --model unsloth/Phi-3-mini-4k-instruct \
  --dataset all'

run_test() {
  local name="$1"
  local args="$2"
  local log="$3"

  echo "▶️  Running $name... Output: $log"

  # Run command, capture both stdout and stderr to the log.
  # Use PIPESTATUS[0] to get the command's exit code (not tee's).
  bash -c "$BASE_CMD $args" 2>&1 | tee "$log"
  local exit_code=${PIPESTATUS[0]}

  if [[ $exit_code -ne 0 ]]; then
    echo "❌ $name failed with exit code $exit_code. See $log"
    {
      echo
      echo "-----"
      echo "[$(date -Is)] $name FAILED (exit $exit_code)"
    } >> "$log"
    exit "$exit_code"
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
LOG1="$LOG_DIR/gpu_test_rag_$(timestamp).txt"
run_test "GPU Test 1 (--rag)" "--rag" "$LOG1"

# Test 2
LOG2="$LOG_DIR/gpu_test_rag_offline_$(timestamp).txt"
run_test "GPU Test 2 (--rag --rag_offline)" "--rag --rag_offline" "$LOG2"

# Test 3
LOG3="$LOG_DIR/gpu_test_no_rag_$(timestamp).txt"
run_test "GPU Test 3 (no --rag)" "" "$LOG3"

echo "🎉 All GPU tests completed successfully."
