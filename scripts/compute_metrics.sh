#!/usr/bin/env bash

usage() {
    echo "Usage: $0 <run_name> [room] [hint_flag]"
    echo
    echo "  run_name     : experiment label / model identifier used in results path"
    echo "  room         : (optional) e.g., room1 (default = all rooms)"
    echo "  hint_flag    : (optional) either 'hint' or 'no_hint' (default = no hint flag)"
    echo
    echo "Example: score_all_runs.sh Qwen3-VL-8B-Instruct"
    echo "         score_all_runs.sh Qwen3-VL-8B-Instruct room5 hint"
    exit 1
}

# Parse input
RUN_NAME="$1"
ROOM_FILTER="$2"
HINT_FLAG="$3"

if [ -z "$RUN_NAME" ]; then
    echo "Error: run_name is required"
    usage
fi

# Determine whether we pass --hint
EVAL_HINT_ARG=""
if [ "$HINT_FLAG" = "hint" ]; then
    EVAL_HINT_ARG="--hint"
fi

# Base results directory
BASE_RESULTS=".scripts/results/Agent"

# Determine rooms
if [ -n "$ROOM_FILTER" ]; then
    # Only score specific room
    ROOMS=("$ROOM_FILTER")
else
    # Score all rooms under both plausible agent types
    # (BaseAgent and VisEscaper)
    mapfile -t ROOMS < <(find "$BASE_RESULTS" -maxdepth 2 -type d \
                          -regex ".*/$RUN_NAME/room[0-9]+" \
                          -printf "%f\n" | sort -u)
fi

if [ ${#ROOMS[@]} -eq 0 ]; then
    echo "No room directories found for run label '$RUN_NAME'"
    exit 1
fi

echo "Scoring for run: $RUN_NAME"
echo "Rooms to evaluate: ${ROOMS[*]}"

# Loop through rooms and scores
for room in "${ROOMS[@]}"; do

    echo
    echo "== Room: $room =="
    
    # A&A: find all trajectories for this run & room
    TRAJ_FILES=$(find "$BASE_RESULTS" -type f \
                 -path "*/$RUN_NAME/$room/trajectory_run_*.json")

    if [ -z "$TRAJ_FILES" ]; then
        echo "No trajectory files found for $RUN_NAME / $room"
        continue
    fi

    # Score each trajectory
    while IFS= read -r traj; do
        echo "  Scoring $traj"
        python evaluation/get_score.py \
            --trajectory "$traj" \
            --room "${room#room}" \
            $EVAL_HINT_ARG
    done <<< "$TRAJ_FILES"
done