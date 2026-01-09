import os
import json
import copy
from datetime import datetime
from typing import Dict, Type
import random
from vis_escape.constants import ASSETS_DIR
from vis_escape.game.manage.game_state import GameState
from vis_escape.game.manage.view_manager import ViewManager
from vis_escape.objects.item import QuizItem
from vis_escape.constants import PROJECT_ROOT
from .agent import Agent
from .. import utils

class AIExperimentRunner:
    def __init__(self, room_name, model_mapping, run_mode, hint_mode, run_name=None):
        self.run_name = run_name
        self.room_name = room_name
        self.model_mapping = model_mapping
        self.run_mode = run_mode
        self.hint_mode = hint_mode
        self.step_count = 0
        self.give_hint_count = 30
        self.ai_player = Agent()
        self.run_history = []
        self.run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_game_state = utils.load_game_state_from_config(os.path.join(ASSETS_DIR, room_name, "config.py"))
        self.view_manager = ViewManager(os.path.join(ASSETS_DIR, room_name), self.model_mapping["caption"], self.current_game_state)
        self.hint_message_dict = self.current_game_state.hint_message if hint_mode == "hint" else {}
        self.previous_game_state = None
        self.previous_message = None
        self.shuffle_action = True
        
        #for AI inputs
        self.previous_scene_path = None
        self.previous_action = None
        self.action_history = []

        #memory
        self.salient_action_history = []
        #plan
        self.current_plan = ""
        self.short_term_memory=[]
        #invalid reflexion
        self.invalid_step_count = 0
        self.currnet_hint_message = ""
        self.given_hints_history=set()
        
    def save_run_history(self):
        utils.save_run_history(
            self.run_history,
            self.room_name,
            self.model_mapping["actor"],
            self.run_mode,
            self.hint_mode,
            self.run_start_time,
            self.current_dir,
            "BaseAgent",
            self.run_name
        )
    def isquiz(self):
        if self.current_game_state.current_view == "ITEM" and self.current_game_state.current_item:
            receptacle = self.current_game_state.get_current_wall().get_receptacle(self.current_game_state.inspected_receptacle)
            if receptacle:
                item_state = receptacle.get_item_state(self.current_game_state.current_item)
                if isinstance(item_state.game_item, QuizItem) and "key" not in item_state.game_item.item_name.lower():
                    return True
        return False
    def get_action_string(self, action, current_location, is_answertry, is_salient):
        if is_salient:
            if is_answertry:
                return f"{current_location}try {action}"
            else:
                return f"{current_location}{action}"
        else:    
            if is_answertry:
                return f"{current_location}try {action}"
            else:
                return action


    def run_experiment(self, max_steps=300):
        experiment_result = {
            "success": False,
            "steps_taken": 0,
            "reason": "incomplete"
        }
        
        is_answertry=False
        while self.step_count < max_steps:
            current_scene_path = self.view_manager.get_current_view_image(self.current_game_state, self.previous_game_state, action=self.previous_action if self.step_count > 0 else None)
            
            current_message = self.view_manager.last_message
            current_location = self.current_game_state.get_current_location()
            available_actions = self.current_game_state.get_available_actions()
            if self.shuffle_action:
                random.shuffle(available_actions)
            inventory = ", ".join(self.current_game_state.context.get_player_inventory_str())
            direction = self.current_game_state.get_current_wall().wall_id
            puzzle = self.isquiz()
            
            if (self.hint_mode=="hint") and (self.invalid_step_count > self.give_hint_count):
                last_trigger, last_trigger_state = self.current_game_state.context.peek_last_trigger()
                if callable(self.current_game_state.hint_message):
                    self.currnet_hint_message = self.current_game_state.hint_message(self.current_game_state)
                else:
                    self.currnet_hint_message = self.hint_message_dict[last_trigger]    
                self.given_hints_history.add(self.currnet_hint_message)
            




            if self.run_mode == "socratic":
                current_scene_desc = current_message["after_state_message"]
            else:  # vlm mode
                current_scene_desc = current_scene_path
                
            if self.step_count == 0:
                try:
                    action, raw_response_action = self.ai_player.get_next_action_first_turn(model=self.model_mapping["actor"],
                        direction=direction,
                        current_scene_desc=current_scene_desc,
                        inventory=inventory,
                        available_actions=available_actions,
                        run_mode=self.run_mode
                    )
                except Exception as e:
                    action = random.choice(available_actions)
                    raw_response_action = f"Error in get_next_action_first_turn: {e}"
            else:
                try:
                    recent_actions = [i for i in self.action_history if "turn" not in i["action"]][-10:]
                    action, raw_response_action, is_answertry = self.ai_player.get_next_action(model=self.model_mapping["actor"],
                        direction=direction,
                        current_scene_desc=current_scene_desc,
                        inventory=inventory,
                        salient_action_history=self.salient_action_history,
                        action_history=recent_actions,
                        previous_action=self.previous_action,
                        available_actions=available_actions,
                        hint_message=self.currnet_hint_message,
                        ispuzzle=puzzle,
                        run_mode=self.run_mode
                    )
                except Exception as e:
                    action = random.choice(available_actions)
                    raw_response_action = f"Error in get_next_action: {e}"
           
            print("\033[94mACTION: ", action, "\033[0m")

            result = self.current_game_state.handle_action(action)
            triggers = self.current_game_state.context.triggers.copy()
            inventory = ", ".join(self.current_game_state.context.get_player_inventory_str())
            
            
                
            reordered_actions = [action] + [act for act in available_actions if act != action]
            salient_action_history_list=[f"Step {i['step']}: {i['action']}" for i in self.salient_action_history[::-1]]
            turn_info = {
                "turn_number": self.step_count,
                "available_actions": ", ".join(
                    f"<{act}>" if act == action else act 
                    for act in reordered_actions
                ),
                "raw_response": raw_response_action,
                "triggers": triggers.copy(),
                "hint_message": self.currnet_hint_message,
                "inventory": inventory,
                "salient_action_history": salient_action_history_list,
                "observation_text": current_message["after_state_message"],
                "observation_image": current_scene_path,
                "given_hints_history": list(self.given_hints_history)
            }

            action_string_salient=self.get_action_string(action, current_location, is_answertry, True)
            action_string_history=self.get_action_string(action, current_location, is_answertry, False)

            self.action_history.append({
                "scene": f"{direction} side of room - {current_message['after_state_message']}",
                "action": action_string_history
            })            
            if "inspect" not in action and "turn" not in action and "step_back" not in action:
                self.salient_action_history.append({
                    "step": self.step_count, 
                    "action": action_string_salient
                })

            self.run_history.append(turn_info)
            self.save_run_history()

            self.step_count += 1
            if self.previous_game_state is not None:
                if self.previous_game_state.context.triggers == self.current_game_state.context.triggers:
                    self.invalid_step_count += 1
                else:
                    self.invalid_step_count = 0
                    self.currnet_hint_message = ""

            self.previous_scene_path = current_scene_path
            self.previous_action = action
            self.previous_game_state = copy.deepcopy(self.current_game_state)
            self.previous_message = copy.deepcopy(current_message)


            if self.current_game_state.check_game_clear():
                experiment_result["success"] = True
                experiment_result["steps_taken"] = self.step_count
                experiment_result["reason"] = "success"
                print(f"Experiment successful! Completed in {self.step_count} steps")
                break
            
            if self.invalid_step_count >= 100:
                experiment_result["reason"] = "too_many_invalid_steps"
                print(f"Experiment failed: Too many invalid steps ({self.invalid_step_count})")
                break

            if self.step_count >= max_steps:
                experiment_result["reason"] = "max_steps_exceeded"
                print(f"Experiment failed: Reached maximum {self.step_count} steps")
                break

        experiment_summary = {
            "experiment_summary": experiment_result
        }
        self.run_history.append(experiment_summary)
        self.save_run_history()
        
        return experiment_result
