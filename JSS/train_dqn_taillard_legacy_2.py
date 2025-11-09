#!/usr/bin/env python3
"""
train_dqn_taillard_single.py
------------------------------------
Train DQN on a *single* Taillard JSSP instance using Ray 1.1.0 / Python 3.8.20.

Usage:
    python3 train_dqn_taillard_single.py --instance instances/ta42 --iters 200 --eval 10
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import gym
from gym import spaces

import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.tune.registry import register_env

# Optional Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ────────────────────────────────────────────────
#  Environment: Taillard Job Shop Scheduling
# ────────────────────────────────────────────────
class TaillardJSSPEnv(gym.Env):
    """Minimal Taillard JSSP environment compatible with Ray 1.1.0 DQNTrainer."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config):
        instance_path = env_config.get("instance_path")
        if instance_path is None:
            raise ValueError("env_config must include 'instance_path'")
        self.instance_path = Path(instance_path)
        self.jobs, self.n_jobs, self.n_machines = self._parse_taillard(self.instance_path)
        self.action_space = spaces.Discrete(self.n_jobs)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_jobs + self.n_machines,), dtype=np.float32
        )
        self.reset()

    def _parse_taillard(self, path):
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        header = lines[0].split()
        n_jobs, n_machines = int(header[0]), int(header[1])
        all_ints = []
        for ln in lines[1:]:
            all_ints += list(map(int, ln.split()))
        jobs, idx = [], 0
        for j in range(n_jobs):
            ops = []
            for m in range(n_machines):
                machine, duration = all_ints[idx], all_ints[idx + 1]
                idx += 2
                ops.append((machine, duration))
            jobs.append(ops)
        return jobs, n_jobs, n_machines

    def reset(self):
        self.job_next = [0] * self.n_jobs
        self.job_end = [0] * self.n_jobs
        self.machine_end = [0] * self.n_machines
        self.done_jobs = [False] * self.n_jobs
        self.total_duration_estimate = sum(sum(d for (_, d) in js) for js in self.jobs)
        return self._get_obs()

    def _get_obs(self):
        job_part = np.array(
            [self.job_next[j] / float(self.n_machines) for j in range(self.n_jobs)],
            dtype=np.float32,
        )
        denom = max(1.0, self.total_duration_estimate)
        mach_part = np.array(
            [self.machine_end[m] / float(denom) for m in range(self.n_machines)],
            dtype=np.float32,
        )
        return np.concatenate([job_part, mach_part], axis=0)

    def step(self, action):
        if action < 0 or action >= self.n_jobs or self.done_jobs[action]:
            return self._get_obs(), -1.0, False, {}

        j = action
        op_idx = self.job_next[j]
        machine, duration = self.jobs[j][op_idx]
        if machine >= self.n_machines:
            machine = machine - 1
        start = max(self.job_end[j], self.machine_end[machine])
        finish = start + duration
        self.job_next[j] += 1
        self.job_end[j] = finish
        self.machine_end[machine] = finish
        if self.job_next[j] >= self.n_machines:
            self.done_jobs[j] = True
        done = all(self.done_jobs)
        if done:
            makespan = max(self.job_end)
            return self._get_obs(), -float(makespan), True, {"makespan": makespan}
        return self._get_obs(), 0.0, False, {}

    def render(self, mode="human"):
        print("Job next:", self.job_next)
        print("Machine end:", self.machine_end)
        print("Job end:", self.job_end)


# ────────────────────────────────────────────────
#  Training Function
# ────────────────────────────────────────────────
def train_instance(instance_path, iters=200, eval_episodes=10, use_wandb=True):
    instance_name = Path(instance_path).name
    results_dir = Path("dqn_legacy_results") / instance_name
    results_dir.mkdir(parents=True, exist_ok=True)

    def env_creator(env_config):
        return TaillardJSSPEnv(env_config)

    register_env("TaillardJSSP", env_creator)

    config = DQN_CONFIG.copy()
    config.update({
        "env": "TaillardJSSP",
        "env_config": {"instance_path": instance_path},
        "num_workers": 0,
        "num_gpus": 0,
        "framework": "tf",
        "train_batch_size": 1024,
        "buffer_size": 200000,
        "target_network_update_freq": 500,
        "lr": 1e-4,
        "gamma": 0.99,
        "learning_starts": 1000,
        "dueling": True,
        "double_q": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 50000,
        },
        "model": {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
    })

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="DQN_JSSP_legacy", name=f"DQN_{instance_name}", reinit=True, config=config)

    ray.init(ignore_reinit_error=True)
    trainer = DQNTrainer(config=config)

    train_log = []
    for i in range(iters):
        result = trainer.train()
        mean_reward = result.get("episode_reward_mean", float("nan"))
        train_log.append({"iter": i + 1, "reward_mean": mean_reward})
        print(f"[{instance_name}] Iter {i + 1}/{iters} - Reward Mean: {mean_reward}")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"iter": i + 1, "reward_mean": mean_reward})

    checkpoint_path = trainer.save(results_dir.as_posix())
    print(f"Checkpoint saved: {checkpoint_path}")

    eval_rows = []
    for ep in range(eval_episodes):
        env = TaillardJSSPEnv({"instance_path": instance_path})
        obs = env.reset()
        done = False
        while not done:
            action = trainer.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
        eval_rows.append({"episode": ep + 1, "makespan": info.get("makespan", None)})
        print(f"[{instance_name}] Eval {ep + 1}: Makespan={info.get('makespan', None)}")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"eval_makespan": info.get("makespan", None)})

    pd.DataFrame(train_log).to_csv(results_dir / "train_log.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(results_dir / "eval_results.csv", index=False)

    trainer.stop()
    ray.shutdown()
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"\n✅ Training complete for {instance_name}. Exiting...\n")


# ────────────────────────────────────────────────
#  CLI Entry Point
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train DQN on a single Taillard instance.")
    parser.add_argument("--instance", type=str, required=True, help="Path to Taillard instance (e.g. instances/ta42)")
    parser.add_argument("--iters", type=int, default=200, help="Training iterations")
    parser.add_argument("--eval", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    if not Path(args.instance).exists():
        raise FileNotFoundError(f"Instance not found: {args.instance}")

    train_instance(args.instance, iters=args.iters, eval_episodes=args.eval, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
