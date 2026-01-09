import json
from pathlib import Path
from typing import Mapping, Optional, Union

import click
import numpy as np

from vis_escape.config.models import get_config, get_preset
from vis_escape.constants import ASSETS_DIR
from vis_escape.experiment.agent.baseagent.experiment_runner import AIExperimentRunner


def check_file(room_name, assets_dir: Optional[Union[str, Path]] = None):
    if not assets_dir:
        assets_dir = Path(ASSETS_DIR)
    mapping_dir = assets_dir / str(room_name) / "mapping"
    image_dir = assets_dir / str(room_name) / "image"
    for viewtype in ["object_view", "wall_view"]:
        view_dir = mapping_dir / viewtype
        
        for mapping_file_path in view_dir.glob("*.json"):
            image_dir_path = image_dir / viewtype / mapping_file_path.stem
            
            with open(mapping_file_path, "r") as f:
                data = json.load(f)
            

def get_model_mapping(
    run_mode: str = "socratic",
    model_name: str = "gpt4o-mini",
    preset_path: Optional[Union[str, Path]] = None,
):
    """
    Get model mapping based on preset configuration.
    
    Args:
        run_mode: "socratic" (VLM for caption, LLM for reasoning) or "vlm" (VLM for all)
        model_name: Preset name from endpoints.yaml (e.g., "gpt4o-mini", "qwen")
        preset_path: Optional custom config path
    
    Returns:
        Dictionary mapping role -> actual model name
    """
    preset = get_preset(model_name, preset_path)
    
    if not preset:
        raise ValueError(f"Preset '{model_name}' not found in configuration")
    
    vlm = preset["vlm"]
    llm = preset["llm"]

    if run_mode == "socratic":
        return {
            "caption": vlm,
            "actor": llm,
            "feedback": llm,
            "memory": llm,
        }
    elif run_mode == "vlm":
        return {
            "caption": vlm,
            "actor": vlm,
            "feedback": vlm,
            "memory": vlm,
        }
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'socratic' or 'vlm'")


def run_multiple_experiments(
    room_name: str,
    num_experiments: int,
    max_steps: int,
    model_mapping: Mapping[str, Mapping[str, str]],
    run_mode: str,
    hint_mode: str,
    run_name: str = None,
):
    print(f"\nRunning {num_experiments} experiments for {room_name}")
    print("-" * 50)

    total_success = 0
    total_steps = 0
    successful_steps = []

    for i in range(num_experiments):
        print(f"\nExperiment {i+1}/{num_experiments}")
        runner = AIExperimentRunner(
            room_name=room_name,
            model_mapping=model_mapping,
            run_mode=run_mode,
            hint_mode=hint_mode,
            run_name=run_name
        )
        result = runner.run_experiment(max_steps=max_steps)

        if result["success"]:
            total_success += 1
            total_steps += result["steps_taken"]
            successful_steps.append(result["steps_taken"])

        print(f"Success: {result['success']}")
        print(f"Steps Taken: {result['steps_taken']}")
        print(f"Reason: {result['reason']}")

    success_rate = (total_success / num_experiments) * 100
    avg_steps = total_steps / total_success if total_success > 0 else 0
    steps_std = np.std(successful_steps) if successful_steps else 0

    print("\nExperiment Summary")
    print("-" * 50)
    print(f"Total Experiments: {num_experiments}")
    print(f"Successful Experiments: {total_success}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Steps (successful runs): {avg_steps:.2f}")
    print(f"Standard Deviation of Steps (successful runs): {steps_std:.2f}")


@click.command()
@click.argument("room_name", type=str)
@click.option(
    "-n",
    "--num-experiments",
    type=int,
    default=3,
    help="Number of experiments to run (default: 3)",
)
@click.option(
    "--max-steps",
    type=int,
    default=200,
    help="Maximum number of steps per experiment (default: 200)",
)
@click.option(
    "--run-name",
    type=str,
    default="default_run",
    help="Name of experiment"
)
@click.option("-m", "--model-name", type=str, default="gpt4o-mini")
@click.option("-t", "--hint-mode", type=str, default="no_hint")
@click.option("-r", "--run-mode", type=str, default="vlm")
def main(room_name, num_experiments, max_steps, model_name, hint_mode, run_mode, run_name):
    """Run AI experiments for room escape."""
    check_file(room_name)
    model_mapping = get_model_mapping(run_mode=run_mode, model_name=model_name)

    run_multiple_experiments(
        room_name,
        num_experiments,
        max_steps,
        model_mapping,
        run_mode,
        hint_mode,
        run_name
    )


if __name__ == "__main__":
    main()
