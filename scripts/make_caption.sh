#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})/..

# Variables
ASSET_ROOT="./assets"

# Parse command line arguments
ROOM_NAME="${1:-room1}"  # Default to room1 if no argument provided

# ===== Choose your configuration =====
# Option 1: OpenAI models
# Uncomment the following lines and set your API key:
# export OPENAI_API_KEY="your-openai-api-key"
# MODEL_NAME="gpt-4o-mini"

# Option 2: vLLM server (local or remote)
# Uncomment the following lines and set your server details:
# MODEL_NAME="llava"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
AGENT_HOSTNAME="localhost"
AGENT_PORT=8000

# Initialize Virtualenvs
conda activate vis-escape

# Build arguments based on configuration
ARGS_STRING="--assets-dir $ASSET_ROOT/$ROOM_NAME --model-name $MODEL_NAME"

# Add vLLM server arguments if configured
if [ ! -z "$AGENT_HOSTNAME" ] && [ ! -z "$AGENT_PORT" ]; then
    ARGS_STRING="$ARGS_STRING --agent-hostname $AGENT_HOSTNAME --agent-port $AGENT_PORT"
    echo "Using vLLM server at $AGENT_HOSTNAME:$AGENT_PORT"
else
    echo "Using OpenAI API"
fi

# Run caption generation
python -m vis_escape.config.caption_item_view $ARGS_STRING
python -m vis_escape.config.caption_wall_view $ARGS_STRING
python -m vis_escape.config.caption_object_view $ARGS_STRING
