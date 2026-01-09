#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Variables
ASSET_ROOT="./assets"

# Parse command line arguments
ROOM_NAME="$1"

# ===== Choose your configuration =====
# Option 1: OpenAI models
# Uncomment the following lines and set your API key:
# export OPENAI_API_KEY="your-openai-api-key"
# MODEL_NAME="gpt-4o-mini"

# Option 2: vLLM server (local or remote)
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
AGENT_HOSTNAME="localhost"
AGENT_PORT=8000

# Activate environment
echo "Activating conda environment 'vis-escape'"
conda activate vis-escape

# Function to run caption generation for a given room
run_caption_for_room() {
    local room="$1"
    local assets_dir="$ASSET_ROOT/$room"

    if [ ! -d "$assets_dir" ]; then
        echo "Skipping '$room': directory does not exist"
        return
    fi

    echo "------------------------------------"
    echo "Generating captions for room: $room"
    echo "Assets directory: $assets_dir"
    echo "------------------------------------"

    # Build base arguments
    ARGS_STRING="--assets-dir $assets_dir --model-name $MODEL_NAME"

    # Add vLLM server arguments if configured
    if [ -n "$AGENT_HOSTNAME" ] && [ -n "$AGENT_PORT" ]; then
        ARGS_STRING="$ARGS_STRING --agent-hostname $AGENT_HOSTNAME --agent-port $AGENT_PORT"
        echo "Using vLLM server at $AGENT_HOSTNAME:$AGENT_PORT"
    else
        echo "Using OpenAI API"
    fi

    # Invoke caption generation
    python -m vis_escape.config.caption_item_view $ARGS_STRING
    python -m vis_escape.config.caption_wall_view $ARGS_STRING
    python -m vis_escape.config.caption_object_view $ARGS_STRING
}

# If a specific room was passed, run only that
if [ -n "$ROOM_NAME" ]; then
    echo "Running caption generation for specific room: $ROOM_NAME"
    run_caption_for_room "$ROOM_NAME"
else
    echo "No specific room provided; running caption generation for all rooms."

    # Iterate all subdirectories in ASSET_ROOT
    for dir in "$ASSET_ROOT"/*; do
        if [ -d "$dir" ]; then
            room="$(basename "$dir")"
            run_caption_for_room "$room"
        fi
    done
fi