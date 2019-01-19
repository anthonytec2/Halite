from ray.rllib.agents.dqn.dqn import DQNAgent, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.agents.dqn import ApexAgent
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
import logging
import os
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gym_halite import env_creator
from model import ParametricActionsModel
CHECKPOINT_FILE = "last_checkpoint.out"

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
        "per_worker_exploration": True,
        "worker_side_prioritization": True,
        "min_iter_time_s": 60,
        "model": {
            "custom_model": "parametric",
            "custom_options": {},  # extra options to pass to your model
        },
        "env_config": {'action': 7,
                       'obs': 1024+7},
    },
)
ray.init(redis_address="localhost:6379")
ModelCatalog.register_custom_model("parametric", ParametricActionsModel)
register_env("halite_env", env_creator)
dqn = ApexAgent(
    env="halite_env",
    config=APEX_DEFAULT_CONFIG)

# Attempt to restore from checkpoint if possible.
if os.path.exists(CHECKPOINT_FILE):
    checkpoint_path = open(CHECKPOINT_FILE).read()
    print("Restoring from checkpoint path", checkpoint_path)
    dqn.restore(checkpoint_path)
    
while True:
    print(pretty_print(dqn.train()))
    checkpoint_path = dqn.save()
    print("Last checkpoint", checkpoint_path)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(checkpoint_path)
