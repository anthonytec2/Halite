import ray
import ray.tune as tune
from ray.rllib.agents.dqn.dqn import DQNAgent, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.agents.dqn import ApexAgent
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
import logging
import os
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gym_halite import env_creator
from model import ParametricActionsModel


APEX_DEFAULT_CONFIG = merge_dicts(
    DQN_CONFIG,  # see also the options in dqn.py, which are also supported
    {
        "optimizer_class": "AsyncReplayOptimizer",
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "max_weight_sync_delay": 400,
                "num_replay_buffer_shards": 4,
                "debug": False
            }),
        "n_step": 3,
        "num_gpus": 1,
        "num_workers": 32,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 512,
        "sample_batch_size": 50,
        "target_network_update_freq": 500000,
        "timesteps_per_iteration": 25000,
        "schedule_max_timesteps": 7500000,
        "exploration_fraction": 0.8,
        "exploration_final_eps": 0.02,
        "per_worker_exploration": True,
        "worker_side_prioritization": True,
        "min_iter_time_s": 60,
        "hiddens": [],
        "lr": 1e-3,
        "model": {
            "custom_model": "parametric",
            "custom_options": {},  # extra options to pass to your model
        },
        "env_config": {'action': 7,
                       'obs': 1024+7,
                       'alpha': .55},
    },
)
ray.init(redis_address="localhost:6379")
ModelCatalog.register_custom_model("parametric", ParametricActionsModel)
register_env("halite_env", env_creator)
tune.run_experiments({
    "my_experiment": {
        "run": ApexAgent,
        "env": "halite_env",
        "config": APEX_DEFAULT_CONFIG,
    },
})