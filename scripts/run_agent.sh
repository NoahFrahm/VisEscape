#!/usr/bin/env bash

ASSET_ROOT="./assets"

DEFAULT_AGENT_TYPE="base"  # base vs visescaper
DEFAULT_MODEL="Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_NUM=5
DEFAULT_HINT="no_hint" #no_hint vs hint
DEFAULT_RUNMODE="vlm"

# Parse args
AGENT_TYPE="${1:-$DEFAULT_AGENT_TYPE}" # "base" or "visescaper"
ROOM_ARG="$2" # specific room or empty for all
MODEL="${3:-$DEFAULT_MODEL}"
NUM_EXPTS="${4:-$DEFAULT_NUM}"
HINT_MODE="${5:-$DEFAULT_HINT}"
RUN_MODE="${6:-$DEFAULT_RUNMODE}"
RUN_NAME="${7:-default_run}"

# Build room list
if [ -n "$ROOM_ARG" ]; then
  ROOMS=("$ROOM_ARG")
else
  mapfile -t ROOMS < <(find "$ASSET_ROOT" -maxdepth 1 -mindepth 1 -type d -printf "%f\n")
fi

echo "Agent will run for rooms: ${ROOMS[*]}"

# Loop over rooms
for room in "${ROOMS[@]}"; do
  echo "Running for room: $room"

  if [ "$AGENT_TYPE" = "base" ]; then
      echo "Running base agent..."
      python scripts/run_baseagent.py "$room" \
          -m "$MODEL" \
          -n "$NUM_EXPTS" \
          -t "$HINT_MODE" \
          -r "$RUN_MODE" \
          -run-name "$RUN_NAME"
  elif [ "$AGENT_TYPE" = "visescaper" ]; then
      echo "Running visescaper agent..."
      python scripts/run_visescaper.py "$room" \
          -m "$MODEL" \
          -n "$NUM_EXPTS" \
          -t "$HINT_MODE" \
          -r "$RUN_MODE" \
          -run-name "$RUN_NAME"
  else
      echo "Error: Unknown agent type '$AGENT_TYPE'. Use 'base' or 'visescaper'."
      exit 1
  fi

done