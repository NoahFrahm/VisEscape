#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <run_name_or_path> [room] [hint_flag]"
    echo
    echo "  run_name_or_path : experiment run name or full path to a VisEscape results run folder"
    echo "  room             : (optional) specific room (e.g., room1). Defaults to all."
    echo "  hint_flag        : (optional) \"hint\" or \"no_hint\" (default no hint)"
    echo
    echo "Examples:"
    echo "  score_all_runs.sh test_run"
    echo "  score_all_runs.sh test_run room5"
    echo "  score_all_runs.sh /full/path/to/run_folder room5 hint"
    exit 1
}

DEFAULT_HINT="no_hint"
DEFAULT_OUTFILE="evaluation_metrics.csv"


# -------------------
# Parse input
# -------------------
RUN_INPUT="$1"
ROOM_FILTER="${2:-}" # fallback to empty if not passed
HINT_FLAG="${3:-$DEFAULT_HINT}"
OUT_FILE="${4:-$DEFAULT_OUTFILE}"

if [ -z "$RUN_INPUT" ]; then
    echo "Error: run_name_or_path is required"
    usage
fi

# Determine hint argument for evaluator
EVAL_HINT_ARG=""
if [ "$HINT_FLAG" = "hint" ]; then
    EVAL_HINT_ARG="--hint"
fi

# Base results root
RESULTS_ROOT="VisEscape/results/VisEscaper/logs"

# Detect run directory
if [ -d "$RUN_INPUT" ]; then
    # If user gave a full results run path
    RUN_DIR="$RUN_INPUT"
else
    # Otherwise assume it lives under the standard results root
    RUN_DIR="$RESULTS_ROOT/$RUN_INPUT"
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Run directory not found: $RUN_DIR"
    exit 1
fi

echo "Using run directory: $RUN_DIR"

# -------------------
# Determine rooms
# -------------------
if [ -n "$ROOM_FILTER" ]; then
    # User specified a single room
    ROOMS=("$ROOM_FILTER")
else
    # Find all room subdirectories under RUN_DIR
    mapfile -t ROOMS < <(find "$RUN_DIR" -maxdepth 1 -type d -name "room*" -printf "%f\n" | sort)
fi

if [ ${#ROOMS[@]} -eq 0 ]; then
    echo "No room directories found under run: $RUN_DIR"
    exit 1
fi

echo "Rooms to evaluate: ${ROOMS[*]}"

# -------------------
# Evaluate trajectories
# -------------------
for room in "${ROOMS[@]}"; do

    ROOM_PATH="$RUN_DIR/$room"
    echo
    echo "== Scoring Room: $room =="

    # Find trajectory JSON files
    mapfile -t TRAJ_FILES < <(find "$ROOM_PATH" -maxdepth 1 -type f -name "run_history_*.json" | sort)

    if [ ${#TRAJ_FILES[@]} -eq 0 ]; then
        echo "No trajectory JSON files found in $ROOM_PATH"
        continue
    fi

    # Loop over each trajectory file
    OUT_PATH="$RUN_DIR/$OUT_FILE"
    for traj in "${TRAJ_FILES[@]}"; do
        echo "Scoring: $traj"
        python -m evaluation.get_score \
            --trajectory "$traj" \
            --room "${room#room}" \
            --out-file "$OUT_PATH" \
            $EVAL_HINT_ARG
    done

done