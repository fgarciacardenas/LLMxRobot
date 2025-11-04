#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

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

# Define base command for readability
BASE_CMD="python3 -m tests.decision_tester.decision_tester \
  --dataset all \
  --ax_local \
  --local_workdir ~/RISCVxLLMxRobot/third_party/voyager-sdk \
  --local_venv ~/RISCVxLLMxRobot/third_party/voyager-sdk/venv/bin/activate \
  --local_run \"export OPENAI_API_KEY=$OPENAI_API_KEY && python3 ~/RISCVxLLMxRobot/third_party/voyager-sdk/inference_llm.py phi3-mini-1024-4core-static\""

# Test 1
LOG1="$LOG_DIR/test_rag_$(date +'%Y%m%d_%H%M%S').txt"
echo "Running Test 1 (with --rag)... Output: $LOG1"
eval "$BASE_CMD --rag" | tee "$LOG1"

# Test 2
LOG2="$LOG_DIR/test_rag_offline_$(date +'%Y%m%d_%H%M%S').txt"
echo "Running Test 2 (with --rag --rag_offline)... Output: $LOG2"
eval "$BASE_CMD --rag --rag_offline" | tee "$LOG2"

# Test 3
LOG3="$LOG_DIR/test_no_rag_$(date +'%Y%m%d_%H%M%S').txt"
echo "Running Test 3 (no --rag)... Output: $LOG3"
eval "$BASE_CMD" | tee "$LOG3"

echo "✅ All tests completed successfully."
