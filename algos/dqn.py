import logging
import os

import ray
from ray.rllib.agents.dqn import DQNAgent
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from gym_halite import env_creator
from model import ParametricActionsModel

CHECKPOINT_FILE = "last_checkpoint.out"


ray.init("localhost:6379")
ModelCatalog.register_custom_model("parametric", ParametricActionsModel)
register_env("halite_env", env_creator)
dqn = DQNAgent(
    env="halite_env",
    config={
        "env_config": {'action': 7,
                       'obs': 1024+7},
        "num_workers": 8,

        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 20,
        "num_gpus": 1,
        "hiddens": [],
        "schedule_max_timesteps": 7500000,
        "timesteps_per_iteration": 4000,
        "exploration_fraction": 0.8,
        "exploration_final_eps": 0.02,
        "lr": 1e-3,
        "model": {
            "custom_model": "parametric",
            "custom_options": {},  # extra options to pass to your model
        }
    })

# Attempt to restore from checkpoint if possible.
if os.path.exists(CHECKPOINT_FILE):
    checkpoint_path = open(CHECKPOINT_FILE).read()
    print("Restoring from checkpoint path", checkpoint_path)
    dqn.restore(checkpoint_path)

# run the new command using the given tracer

# make a report, placing output in the current directory

# Serving and training loop
while True:
    print(pretty_print(dqn.train()))
    checkpoint_path = dqn.save()
    print("Last checkpoint", checkpoint_path)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(checkpoint_path)
