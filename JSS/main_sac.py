#!/usr/bin/env python3
"""
main_sac.py
-----------
Standalone Soft Actor-Critic (SAC) training script for Job Shop Scheduling
Compatible with Python 3.8 and Ray RLlib.

Trains sequentially on Taillard instances: ta42, ta52, ta62, ta72
and logs results to Weights & Biases (W&B).
"""

import os
import time
import random
import numpy as np
import multiprocessing as mp
import ray
import wandb
import ray.tune.integration.wandb as wandb_tune

from typing import Dict, Tuple
from ray.tune.utils import flatten_dict
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf

from CustomCallbacks import CustomCallbacks
from models import FCMaskedActionsModelTF

# Ensure TensorFlow import (Ray uses TF1.x for some parts)
tf1, tf, tfv = try_import_tf()

# --- W&B field filters ---
_exclude_results = ["done", "should_checkpoint", "config"]
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id",
    "hostname", "pid", "date",
]


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    """Format RLlib result for logging to wandb."""
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(k.startswith(item + "/") or k == item for item in _config_results):
            config_update[k] = v
        elif any(k.startswith(item + "/") or k == item for item in _exclude_results):
            continue
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)
    return log, config_update


def train_sac():
    """Train SAC model sequentially on multiple Taillard instances."""
    instances = ["instances/ta42", "instances/ta52", "instances/ta62", "instances/ta72"]

    # ==== Base SAC Configuration ====
    config = {
        "env": "JSSEnv:jss-v1",
        "seed": 0,
        "framework": "tf",
        "log_level": "WARN",
        "num_workers": min(mp.cpu_count(), 8),
        "num_gpus": 1,
        "gamma": 0.99,
        "tau": 0.005,
        "train_batch_size": 256,
        "target_entropy": "auto",
        "optimization": {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        },
        "num_steps_sampled_before_learning_starts": 1000,
        "learning_starts": 1000,
    }

    wandb.init(project="JSS", config=config)
    ray.init(ignore_reinit_error=True)

    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Custom model for action masking
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config["model"] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    }

    config = with_common_config(config)
    config["callbacks"] = CustomCallbacks

    # --- Train each instance sequentially ---
    stop = {"time_total_s": 180 * 60}  # 3 hours max per instance

    for instance_path in instances:
        print(f"\n=== Training SAC on {instance_path} ===")
        config["env_config"] = {"env_config": {"instance_path": instance_path}}

        trainer = SACTrainer(config=config)

        start_time = time.time()
        while time.time() - start_time < stop["time_total_s"]:
            result = trainer.train()
            result = wandb_tune._clean_log(result)
            log, config_update = _handle_result(result)
            wandb.log(log)

        # Save checkpoint per instance
        checkpoint = trainer.save()
        print(f"✅ Checkpoint saved for {instance_path}: {checkpoint}")

    ray.shutdown()
    print("\nAll SAC training runs completed!")


if __name__ == "__main__":
    train_sac()
