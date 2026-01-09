import json
import os
from pathlib import Path
from typing import Mapping, Optional

import click

from vis_escape.utils import run_inference_vision_caption



def update_captions_json(
    captions_file: str, item_name: str, image_path: str, caption: str
):
    """Update the shared captions JSON file under the IV (Item View) section"""
    captions = {}
    if os.path.exists(captions_file):
        with open(captions_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

    # Initialize nested dictionaries if they don't exist
    captions.setdefault("item_view", {})
    captions["item_view"].setdefault(item_name, {})

    filename = os.path.basename(image_path)
    captions["item_view"][item_name][filename] = caption

    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def process_item_view_json(
    json_path: str,
    assets_dir: str,
    captions_file: str,
    model_name: str,
    base_url: str,
):
    """Process a single item view JSON file and generate captions for its states"""
    item_name = Path(json_path).stem  # Get filename without extension

    with open(json_path, "r", encoding="utf-8") as f:
        item_data = json.load(f)

    for state in item_data["item_states"]:
        # Add item_name folder to the path
        image_path = os.path.join(
            assets_dir, "image", "item_view", item_name, state["image_path"]
        )

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue

        # Create prompt based on state information
        state_name = state["name"]

        prompt = f"""This image is a close-up view of an item '{item_name}'.
Describe this image. Your description should fulfill the following rules:
1. Description should include every visual information, but concise and clear.
2. Do not start description with phrases like 'The image depicts', 'The image shows', etc."""

        print(f"\nProcessing {state['image_path']}...")
        print(f"Prompt: {prompt}")
        caption = run_inference_vision_caption(
            image_path, prompt, model_name, base_url
        )
        print(f"Generated caption: {caption}")

        update_captions_json(captions_file, item_name, state["image_path"], caption)


@click.command()
@click.option("-a", "--assets-dir", type=str, required=True)
@click.option("-m", "--model-name", type=str, required=True)
@click.option("-u", "--agent-hostname", type=str, required=False, default=None)
@click.option("-p", "--agent-port", type=int, required=False, default=None)
def main(assets_dir, model_name, agent_hostname, agent_port):
    assets_dir = Path(assets_dir).expanduser().absolute()

    item_view_dir = assets_dir / "mapping" / "item_view"
    if not item_view_dir.exists():
        raise ValueError(f"Directory not found: {item_view_dir}")

    captions_file = assets_dir / "captions" / model_name / f"image_captions.json"
    captions_file.parent.mkdir(parents=True, exist_ok=True)

    # Get base url for vLLM (None if using OpenAI)
    base_url = f"http://{agent_hostname}:{agent_port}/v1" if agent_hostname and agent_port else None

    # Process each item view JSON file
    for json_path in item_view_dir.glob("*.json"):
        print(f"\nProcessing item view file: {json_path.name}")
        process_item_view_json(
            str(json_path),
            str(assets_dir),
            str(captions_file),
            model_name,
            base_url,
        )

    print(f"\nItem view captions have been saved to {captions_file}")


if __name__ == "__main__":
    main()
