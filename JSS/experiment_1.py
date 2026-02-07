import os
import json
import gym
import pandas as pd
import numpy as np
from models import *
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import ray

# Assuming your custom model class is available in your workspace
# from your_module import FCMaskedActionsModelTF 

if not ray.is_initialized():
    ray.init()

def load_ppo_model(checkpoint_path, params_path):
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
    with open(params_path, "r") as f:
        config = json.load(f)
    
    if "callbacks" in config:
        del config["callbacks"]
        
    config["num_workers"] = 0 
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    return trainer

def solve_instance(trainer, instance_id):
    # Path construction based on your default_config.py structure
    instance_path = f"instances/{instance_id}"
    env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': instance_path})
    
    state = env.reset()
    done = False
    while not done:
        # Pass the observation and the action mask to the trainer
        action = trainer.compute_action(state, explore=False)
        state, reward, done, _ = env.step(action)
    
    # Extract makespan from the environment as seen in FIFO.py
    makespan = env.last_time_step
    return makespan

# Best Known Solutions (BKS) for Taillard 20x20, 30x15, 30x20, and 50x15 instances
# Data sourced from the OR-Library (Beasley) and Taillard's original benchmarks.
# https://scheduleopt.github.io/benchmarks/jsplib/#best-known-solutions---jsplib
bks_data = {
    "ta41": 2005, "ta42": 1937, "ta43": 1846, "ta44": 1979, "ta45": 1997,
    "ta46": 2004, "ta47": 1889, "ta48": 1937, "ta49": 1960, "ta50": 1923,
    "ta51": 2760, "ta52": 2756, "ta53": 2717, "ta54": 2839, "ta55": 2679,
    "ta56": 2781, "ta57": 2943, "ta58": 2885, "ta59": 2655, "ta60": 2723,
    "ta61": 2868, "ta62": 2869, "ta63": 2755, "ta64": 2702, "ta65": 2725,
    "ta66": 2845, "ta67": 2825, "ta68": 2784, "ta69": 3071, "ta70": 2995,
    "ta71": 5464, "ta72": 5181, "ta73": 5568, "ta74": 5339, "ta75": 5392,
    "ta76": 5342, "ta77": 5436, "ta78": 5394, "ta79": 5358, "ta80": 5183
}

model_folders = ["PPO_ta42_checkpoint_27012026", "PPO_ta52_checkpoint_28012026", "PPO_ta62_checkpoint_30012026", "PPO_ta72_checkpoint_31012026"]
base_dir = "checkpoint_results"
all_results = []

for folder in model_folders:
    checkpoint_path = os.path.join(base_dir, folder, "checkpoint_300", "checkpoint-300")
    params_path = os.path.join(base_dir, folder, "params.json")
    
    print(f"Evaluating model trained on {folder}...")
    trainer = load_ppo_model(checkpoint_path, params_path)
    
    # Determine target instance range (e.g., ta42 model evaluates ta41-ta50)
    group_num = int(folder[6]) # extract '4' from 'ta42'
    instance_range = [f"ta{i}" for i in range(group_num * 10 + 1, group_num * 10 + 11)]
    
    for inst in instance_range:
        achieved = solve_instance(trainer, inst)
        bks = bks_data[inst]
        all_results.append({
            "Model Group": f"{folder} [ta{group_num*10+1}-ta{group_num*10+10}]",
            "Instance": inst,
            "BKS": bks,
            "Achieved Result": achieved,
            "Gap (%)": round(((achieved - bks) / bks) * 100, 2)
        })

# Create and export Table
df = pd.DataFrame(all_results)
df.to_csv("tabulated_ppo_results.csv", index=False)
print("Evaluation complete. Results saved to tabulated_ppo_results.csv.")
