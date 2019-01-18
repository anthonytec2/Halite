import logging
import os

import ray
from ray.rllib.agents.dqn import DQNAgent
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from algos.gym_halite import env_creator
from algos.model import ParametricActionsModel

CHECKPOINT_FILE = "last_checkpoint.out"

ray.init(local_mode=True)
ModelCatalog.register_custom_model("parametric", ParametricActionsModel)
register_env("halite_env", env_creator)
dqn = DQNAgent(
    env="halite_env",
    config={
        "env_config": {},
        "num_workers": 1,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "num_gpus": 1,
        "hiddens": [],
        "schedule_max_timesteps": 100000000,
        "timesteps_per_iteration": 1000,
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

env1 = env_creator({})
obs = env1.reset()
done = False
while not done:
    obs, reward, done, _ = env1.step(dqn.compute_action(obs))
