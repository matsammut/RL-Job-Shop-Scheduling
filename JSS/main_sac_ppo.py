import time
import ray
import wandb
import random
import numpy as np
import ray.tune.integration.wandb as wandb_tune
from typing import Dict, Tuple
import multiprocessing as mp

from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf

# === Import RLlib algorithms ===
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer

from CustomCallbacks import *
from models import *

tf1, tf, tfv = try_import_tf()

_exclude_results = ["done", "should_checkpoint", "config"]

_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
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


def train_func():
    # ==== Choose algorithm: "PPO" or "SAC" ====
    ALGORITHM = "SAC"  # change to "PPO" or "SAC"

    instances = ["instances/ta42", "instances/ta52", "instances/ta62", "instances/ta72"]

    # ==== Shared Base configuration ====
    base_config = {
        "env": "JSSEnv:jss-v1",
        "seed": 0,
        "framework": "tf",
        "log_level": "WARN",
        "num_workers": min(mp.cpu_count(), 8),
        "num_gpus": 1,
    }

    if ALGORITHM == "PPO":
        config = base_config.copy()
        config.update({
            "train_batch_size": 4000,
            "rollout_fragment_length": 704,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,  # epochs
            "clip_param": 0.5,
            "vf_loss_coeff": 0.8,
            "kl_coeff": 0.5,
            "lambda": 1.0,
            "entropy_start": 2.0e-3,
            "entropy_end": 2.5e-4,
            "lr_start": 6.6e-4,
            "lr_end": 7.8e-5,
            "gamma": 1.0,
            "batch_mode": "truncate_episodes",
            "shuffle_sequences": True,
            "observation_filter": "NoFilter",
        })
    else:  # SAC configuration for Ray 1.1.0
        config = base_config.copy()
        config.update({
            "gamma": 0.99,
            "tau": 0.005,
            "train_batch_size": 2096,
            "learning_starts": 1000,
            "buffer_size": int(1e6),
            "target_entropy": -3.0,   # must be numeric, not 'auto'
            "use_state_preprocessor": False,
            "no_done_at_end": False,
            "optimization": {
                "actor_learning_rate": 3e-4,
                "critic_learning_rate": 3e-4,
                "entropy_learning_rate": 3e-4,
            },
	    "Q_model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [256, 256],
            },
        })
    wandb.init(project="JSS", config=config)
    ray.init(ignore_reinit_error=True)
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config["model"] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    }

    config = with_common_config(config)
    config["callbacks"] = CustomCallbacks

    # PPO-specific learning rate + entropy schedules
    if ALGORITHM == "PPO":
        config["lr"] = config["lr_start"]
        config["lr_schedule"] = [
            [0, config["lr_start"]],
            [1_000_000, config["lr_end"]],
        ]
        config["entropy_coeff"] = config["entropy_start"]
        config["entropy_coeff_schedule"] = [
            [0, config["entropy_start"]],
            [1_000_000, config["entropy_end"]],
        ]

    stop = {"time_total_s": 180 * 60}  # 3 hours per instance

    for instance_path in instances:
        print(f"\n=== Training on {instance_path} using {ALGORITHM} ===")
        config["env_config"] = {"env_config": {"instance_path": instance_path}}

        if ALGORITHM == "PPO":
            trainer = PPOTrainer(config=config)
        else:
            trainer = SACTrainer(config=config)

        start_time = time.time()
        while time.time() - start_time < stop["time_total_s"]:
            result = trainer.train()
            result = wandb_tune._clean_log(result)
            log, config_update = _handle_result(result)
            wandb.log(log)

        checkpoint = trainer.save()
        print(f"Checkpoint saved for {instance_path}: {checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    train_func()
