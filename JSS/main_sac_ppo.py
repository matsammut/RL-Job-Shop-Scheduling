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
    ALGORITHM = "SAC"  # change to "SAC" to run Soft Actor-Critic

    instances = ["instances/ta42", "instances/ta52", "instances/ta62", "instances/ta72"]

    # ==== Base configuration ====
    default_config = {
        'env': 'JSSEnv:jss-v1',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'num_envs_per_worker': 4,
        'batch_mode': "truncate_episodes",
        'shuffle_sequences': True,
        'observation_filter': "NoFilter",
        'simple_optimizer': False,
        '_fake_gpus': False,
        'num_gpus': 1,
    }

    # PPO-specific parameters
    if ALGORITHM == "PPO":
        default_config.update({
            'train_batch_size': 4000,
            'rollout_fragment_length': 704,
            'sgd_minibatch_size': 128,
            'num_sgd_iter': 10,  # epochs
            'clip_param': 0.5,
            'vf_loss_coeff': 0.8,
            'kl_coeff': 0.5,
            'lambda': 1.0,
            'entropy_start': 2.0e-3,
            'entropy_end': 2.5e-4,
            'lr_start': 6.6e-4,
            'lr_end': 7.8e-5,
        })
    elif ALGORITHM == "SAC":
        # SAC typical parameters
        default_config.update({
            "tau": 0.005,
            "train_batch_size": 256,
            "target_entropy": "auto",
            "optimization": {
                "actor_learning_rate": 3e-4,
                "critic_learning_rate": 3e-4,
                "entropy_learning_rate": 3e-4,
            },
        })

    wandb.init(project="JSS", config=default_config)
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    }

    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks

    # PPO schedules
    if ALGORITHM == "PPO":
        config['lr'] = config['lr_start']
        config['lr_schedule'] = [
            [0, config['lr_start']],
            [1_000_000, config['lr_end']],
        ]
        config['entropy_coeff'] = config['entropy_start']
        config['entropy_coeff_schedule'] = [
            [0, config['entropy_start']],
            [1_000_000, config['entropy_end']],
        ]

    stop = {"time_total_s": 200 * 60}  # 200 minutes per instance

    for instance_path in instances:
        print(f"\n=== Training on {instance_path} with {ALGORITHM} ===")
        config['env_config'] = {'env_config': {'instance_path': instance_path}}

        trainer = PPOTrainer(config=config) if ALGORITHM == "PPO" else SACTrainer(config=config)

        start_time = time.time()
        while start_time + stop['time_total_s'] > time.time():
            result = trainer.train()
            result = wandb_tune._clean_log(result)
            log, config_update = _handle_result(result)
            wandb.log(log)

        # save checkpoint
        checkpoint = trainer.save()
        print(f"Checkpoint saved for {instance_path}: {checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    train_func()
