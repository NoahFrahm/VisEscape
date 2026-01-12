#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="./assets"

# Default values
DEFAULT_RUN_NAME="default_run"
DEFAULT_AGENT_TYPE="base"
DEFAULT_MODEL="qwen3"
DEFAULT_HINT="no_hint"
DEFAULT_RUNMODE="vlm"
DEFAULT_NUM=1

# Parse positional args in preferred order
RUN_NAME="${1:-$DEFAULT_RUN_NAME}"        # most likely to change
AGENT_TYPE="${2:-$DEFAULT_AGENT_TYPE}"    # base | visescaper
MODEL="${3:-$DEFAULT_MODEL}"              # model name
HINT_MODE="${4:-$DEFAULT_HINT}"           # hint | no_hint
RUN_MODE="${5:-$DEFAULT_RUNMODE}"         # vlm | socratic
NUM_EXPTS="${6:-$DEFAULT_NUM}"            # number of experiments
ROOM_ARG="${7:-}"                         # specific room (optional)

# Build room list
if [ -n "$ROOM_ARG" ]; then
  ROOMS=("$ROOM_ARG")
else
  mapfile -t ROOMS < <(find "$ASSET_ROOT" -maxdepth 1 -mindepth 1 -type d -printf "%f\n")
fi

echo "=== Running with configuration ==="
echo "Run name   : $RUN_NAME"
echo "Agent type : $AGENT_TYPE"
echo "Model      : $MODEL"
echo "Hint mode  : $HINT_MODE"
echo "Run mode   : $RUN_MODE"
echo "Num runs   : $NUM_EXPTS"
echo "Rooms      : ${ROOMS[*]}"
echo "=================================="
echo

# Loop over rooms
for room in "${ROOMS[@]}"; do
  echo ">>> Starting agent for room: $room"

  if [ "$AGENT_TYPE" = "base" ]; then
      echo "Running BaseAgent..."
      python scripts/run_baseagent.py "$room" \
          -m "$MODEL" \
          -n "$NUM_EXPTS" \
          -t "$HINT_MODE" \
          -r "$RUN_MODE" \
          --run-name "$RUN_NAME"
  elif [ "$AGENT_TYPE" = "visescaper" ]; then
      echo "Running VisEscaper..."
      python scripts/run_visescaper.py "$room" \
          -m "$MODEL" \
          -n "$NUM_EXPTS" \
          -t "$HINT_MODE" \
          -r "$RUN_MODE" \
          --run-name "$RUN_NAME"
  else
      echo "Error: Unknown agent type '$AGENT_TYPE'. Use 'base' or 'visescaper'."
      exit 1
  fi

  echo
done