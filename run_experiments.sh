#!/bin/bash
# Run all stigma experiments: 8 models x 3 modes = 24 combinations.
# Experiments run sequentially; use --workers to parallelize calls within each run.
# Usage: bash run_experiments.sh [--limit N] [--workers N]

set -u

# Run from the repo root so relative paths resolve correctly
cd "$(dirname "$0")"

LIMIT_ARG=""
WORKERS_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)  LIMIT_ARG="--limit $2"; shift 2 ;;
    --workers) WORKERS_ARG="--workers $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; echo "Usage: bash run_experiments.sh [--limit N] [--workers N]"; exit 1 ;;
  esac
done

MODELS=(
  "us.anthropic.claude-opus-4-5-20251101-v1:0"
  "us.anthropic.claude-sonnet-4-20250514-v1:0"
  "us.meta.llama3-3-70b-instruct-v1:0"
  "deepseek.v3-v1:0"
  "openai.gpt-oss-120b-1:0"
  "openai.gpt-oss-20b-1:0"
  "us.meta.llama4-maverick-17b-instruct-v1:0"
  "us.meta.llama4-scout-17b-instruct-v1:0"
)

PROMPTS="data/prompts_reduced.jsonl"
OUTPUT="results"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

FAILED=0

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME=$(echo "$MODEL" | tr '/:.' '_')

  for MODE in "plain" "cot" "ntcot"; do
    MODE_FLAG=""
    [ "$MODE" == "cot" ] && MODE_FLAG="--use-cot"
    [ "$MODE" == "ntcot" ] && MODE_FLAG="--ntcot"

    LOG_FILE="$LOG_DIR/${SAFE_NAME}_${MODE}.log"
    echo "Running: $MODEL - $MODE -> $LOG_FILE"

    if ! python src/stigma.py --model "$MODEL" \
      --prompts-file "$PROMPTS" --output-directory "$OUTPUT" \
      $MODE_FLAG $LIMIT_ARG $WORKERS_ARG \
      > "$LOG_FILE" 2>&1; then
      echo "  FAILED: $MODEL - $MODE (see $LOG_FILE)"
      ((FAILED++))
    fi
  done
done

echo ""
if [ $FAILED -eq 0 ]; then
  echo "All experiments completed successfully!"
else
  echo "$FAILED experiment(s) failed. Check logs in $LOG_DIR/"
  exit 1
fi
