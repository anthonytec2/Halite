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
from ray.tune.schedulers import PopulationBasedTraining
import random

config={
        "env_config": {'action': 7,
                       'obs': 1024+7,
                       'alpha': .55},
        "num_workers": 31,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 20,
        "num_gpus": 1,
        "hiddens": [],
        "schedule_max_timesteps": 7500000,
        "timesteps_per_iteration": 4000,
        "exploration_fraction": 0.8,
        "exploration_final_eps": 0.02,
        "lr": 1e-3,
        "train_batch_size": 512,
        "model": {
            "custom_model": "parametric",
            "custom_options": {},  # extra options to pass to your model
        }
    }
ray.init(redis_address="localhost:6379")
ModelCatalog.register_custom_model("parametric", ParametricActionsModel)
register_env("halite_env", env_creator)

pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr='episode_reward_mean',
        perturbation_interval=64,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4],
            "train_batch_size": [128, 256],
        })
tune.run_experiments({
    "my_experiment": {
        "run": DQNAgent,
        "env": "halite_env",
        "config": config,
        "num_samples":4,
        #"stop": {"timesteps_total": 7500000},
        #"resources_per_trial": {"cpu": 32, "gpu": 1},
        #"checkpoint_freq": 5 ,
        #"checkpoint_at_end": True,
        
    },
}, queue_trials=True, scheduler=pbt_scheduler)