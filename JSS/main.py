import time

import ray
import wandb

import random

import numpy as np

import ray.tune.integration.wandb as wandb_tune

from ray.rllib.agents.ppo import PPOTrainer

from CustomCallbacks import *
from models import *

from typing import Dict, Tuple

import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog

from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


_exclude_results = ["done", "should_checkpoint", "config"]

# Use these result keys to update `wandb.config`
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(
                k.startswith(item + "/") or k == item
                for item in _config_results):
            config_update[k] = v
        elif any(
                k.startswith(item + "/") or k == item
                for item in _exclude_results):
            continue
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update


def train_func():
    default_config = {
        'env': 'JSSEnv:jss-v1',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 1,
        'instance_path': 'instances/ta52',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'train_batch_size':4000,
        'num_envs_per_worker': 4,
        'rollout_fragment_length': 704,  # TO TUNE
        'sgd_minibatch_size': 128,
        'num_sgd_iter': 10,          # epochs
	'clip_param': 0.5,
        'vf_loss_coeff': 0.8,
        'kl_coeff': 0.5,
        'lambda': 1.0,
        'entropy_start': 2.0e-3,
        'entropy_end': 2.5e-4,
        'lr_start': 6.6e-4,
        'lr_end': 7.8e-5,
        "batch_mode": "truncate_episodes",
        "grad_clip": None,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "vf_share_layers": False,
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,
    }

    wandb.init(config=default_config)
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
    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
    }

    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks


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
    config.pop('instance_path', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    stop = {
        "time_total_s": 200 * 60,
    }

    start_time = time.time()
    trainer = PPOTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)

    ray.shutdown()


if __name__ == "__main__":
    train_func()
