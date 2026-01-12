import json
import csv
import pandas as pd
import os
import argparse
from typing import Dict
from src.vis_escape.utils import ensure_csv_ext

ORACLE_STEPS = {
    1: 28, 2: 26, 3: 31, 4: 39, 5: 36, 6: 23, 7: 22, 8: 31, 9: 33, 10: 28,
    11: 22, 12: 22, 13: 21, 14: 23, 15: 23, 16: 23, 17: 29, 18: 22, 19: 30, 20: 32
}

CLEAR_PROGRESS = {
    1: 10, 2: 10, 3: 10, 4: 12, 5: 10, 6: 6, 7: 6, 8: 10, 9: 10, 10: 9,
    11: 6, 12: 7, 13: 6, 14: 6, 15: 7, 16: 8, 17: 8, 18: 7, 19: 9, 20: 10
}


def get_solved_hints_num(hint_messages: list) -> int:
    num = 0
    hint_received = False
    
    for hint_message in hint_messages:
        if hint_message != "":
            hint_received = True
        else:
            if hint_received:
                num += 1
                hint_received = False
    return num


def get_hint_type_num(hints: list) -> tuple:
    quiz_hint_num = 0
    action_hint_num = 0
    
    for hint in hints:
        if "answer" in hint.lower():
            quiz_hint_num += 1
        else:
            action_hint_num += 1
    
    return quiz_hint_num, action_hint_num


def analyze_trajectory(trajectory_file: str, room_number: int, hint_mode: str = "no_hint") -> Dict:
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    if len(data) < 2:
        raise ValueError("Invalid trajectory format: needs at least 2 entries")
    
    success = "experiment_summary" in data[-1] and data[-1]["experiment_summary"]["success"]
    steps = data[-1]["experiment_summary"]["steps_taken"] if success else 300
    progress = len(data[-2].get("triggers", []))
    
    goal_completion = progress / CLEAR_PROGRESS[room_number]
    
    spl = ORACLE_STEPS[room_number] / steps if success else 0.0
    
    result = {
        "success": success,
        "steps": steps,
        "progress": progress,
        "goal_completion_ratio": goal_completion,
        "spl": spl
    }
    
    if hint_mode == "hint":
        given_hints = data[-2].get("given_hints_history", [])
        hint_messages = [item.get("hint_message", "") for item in data[:-1]]
        
        given_hints_num = len(given_hints)
        solved_hints_num = get_solved_hints_num(hint_messages)
        quiz_hint_num, action_hint_num = get_hint_type_num(given_hints)
        
        total_hints = action_hint_num + quiz_hint_num
        action_hint_ratio = action_hint_num / total_hints if total_hints > 0 else 0.0
        
        result.update({
            "given_hints_num": given_hints_num,
            "solved_hints_num": solved_hints_num,
            "quiz_hint_num": quiz_hint_num,
            "action_hint_num": action_hint_num,
            "action_hint_ratio": action_hint_ratio
        })
    
    return result


def print_metrics(metrics: Dict):
    print("\n=== Trajectory Analysis ===")
    print(f"Success: {metrics['success']}")
    print(f"Steps Taken: {metrics['steps']}")
    print(f"Progress: {metrics['progress']}")
    print(f"Goal Completion Ratio: {metrics['goal_completion_ratio']:.2%}")
    print(f"SPL: {metrics['spl']:.2%}")
    
    if "given_hints_num" in metrics:
        print("\n=== Hint Metrics ===")
        print(f"Given Hints: {metrics['given_hints_num']}")
        print(f"Solved Hints: {metrics['solved_hints_num']}")
        print(f"Quiz Hints: {metrics['quiz_hint_num']}")
        print(f"Action Hints: {metrics['action_hint_num']}")
        print(f"Action Hint Ratio: {metrics['action_hint_ratio']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a single trajectory file')
    parser.add_argument('--trajectory', type=str, required=True,
                        help='Path to the trajectory JSON file')
    parser.add_argument('--room', type=int, required=True,
                        help='Room number (1-20)')
    parser.add_argument('--hint', action='store_true',
                        help='Enable hint mode analysis')
    parser.add_argument('--out-file', required=True,
                        help='file path to save metrics to')
    
    args = parser.parse_args()
    
    hint_mode = "hint" if args.hint else "no_hint"
    
    metrics = analyze_trajectory(args.trajectory, args.room, hint_mode)
    metrics['trajectory_name'] = args.trajectory
    metrics['room'] = args.room

    print_metrics(metrics)
    # breakpoint()

    df = pd.DataFrame([metrics])
    csv_path = ensure_csv_ext(args.out_file)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False,
        columns=["room", "success","steps", "progress", "goal_completion_ratio", "spl", "trajectory_name"],
    )


