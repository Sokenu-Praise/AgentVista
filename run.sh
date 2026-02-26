#!/bin/bash
# Run greedy tool-calling agent. Set env vars before running (see README).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Required: set these in the environment or in a .env file
export REASONING_MODEL_NAME="${REASONING_MODEL_NAME:-openai/gpt-4o}"
export REASONING_API_KEY="${REASONING_API_KEY:?Error: REASONING_API_KEY is not set}"
export REASONING_END_POINT="${REASONING_END_POINT:-https://openrouter.ai/api/v1/chat/completions}"

export SERPAPI_KEY="${SERPAPI_KEY:?Error: SERPAPI_KEY is not set}"
export JINA_API_KEY="${JINA_API_KEY:?Error: JINA_API_KEY is not set}"

# Optional: Verifier for automatic accuracy scoring (requires ground truth in input data)
export VERIFIER_MODEL_NAME="${VERIFIER_MODEL_NAME:-openai/gpt-4.1}"
export VERIFIER_API_KEY="${VERIFIER_API_KEY:-$REASONING_API_KEY}"
export VERIFIER_END_POINT="${VERIFIER_END_POINT:-$REASONING_END_POINT}"

# Optional: tool list (default from env ENABLED_TOOLS or all)
# export ENABLED_TOOLS="web_search,image_search,visit,code_interpreter"

# Paths (override as needed)
# Default paths point to datasets folder in parent directory
INPUT_FILE="${INPUT_FILE:-../datasets/val.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

mkdir -p "$OUTPUT_DIR"

python infer.py \
  --input-file "$INPUT_FILE" \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir "$OUTPUT_DIR" \
  --max-turns 30 \
  --max-images 100 \
  --max-total-tokens 65536 \
  --skip-completed \
  "$@"

echo "Done. Results in $OUTPUT_DIR"
