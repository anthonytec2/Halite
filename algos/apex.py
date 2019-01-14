from __future__ import absolute_import, division, print_function

import os
import gym
import numpy as np
import psutil
import ray
import tensorflow as tf
import tensorflow.contrib.slim as slim
import zmq
from gym import spaces
from ray.rllib.agents.dqn import ApexAgent
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.misc import get_activation_fn, normc_initializer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import logging

CHECKPOINT_FILE = "last_checkpoint.out"


def env_creator(env_config):
    return HaliteEnv({})


class HaliteEnv(gym.Env):

    def __init__(self, config):
        input_obs = spaces.Dict({
            "action_mask": spaces.Box(low=0, high=1, shape=(7, ), dtype=np.float32),
            "real_obs": spaces.Box(low=0, high=1, shape=(27, ), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(7)
        self.observation_space = input_obs
        self.establish_conn()

    def step(self, action):
        act_msg = str(action)
        self.socket.send(act_msg.encode("utf-8"))
        try:
            data = self.socket.recv()

            res = np.frombuffer(data, dtype=np.float32)
            obs = res[:27]
            mask = res[27:34]
            reward = res[-2]
            done = res[-1]
            done = True if done == 1 else False
        except:
            obs = np.zeros(27)
            mask = np.ones(7)
            mask[-1] = 0
            mask[-2] = 0
            reward = 0
            done = True

        obs_ret = {
            "action_mask": mask,
            "real_obs": obs,
        }
        return obs_ret, reward, done, {}

    def reset(self):
        self.run_program()
        data = self.socket.recv()
        res = np.frombuffer(data, dtype=np.float32)
        obs = res[:27]
        mask = res[27:34]
        obs_ret = {
            "action_mask": mask,
            "real_obs": obs,
        }
        return obs_ret

    def establish_conn(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.rnd_num = str(np.random.random())
        self.socket.bind("ipc:///tmp/v{}".format(self.rnd_num))
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)

    def run_program(self):
        cmd = '/home/abisulco/Halite/halite --replay-directory replays/ -vvv --width 32 --height 32 --no-timeout --no-logs --no-replay "python3.6 /home/abisulco/Halite/bots/networking.py --port={}" "python3.6 /home/abisulco/Halite/bots/Bot2.py" &'.format(
            self.rnd_num)
        self.res = psutil.Popen(
            cmd, shell=True)


class ParametricActionsModel(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):
        action_mask = input_dict["obs"]["action_mask"]
        obs = input_dict["obs"]["real_obs"]
        # Standard FC net component.
        last_layer = obs
        hiddens = [20, 20, 15, 15, 10, 10, 9, 9]
        # hiddens = [256, 256]
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = slim.fully_connected(
                last_layer,
                size,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.tanh,
                scope=label)
        output = slim.fully_connected(
            last_layer,
            7,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="fc_out")
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        ouput_mask = output+inf_mask
        return ouput_mask, last_layer


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
        "min_iter_time_s": 30,
        "model": {
            "custom_model": "parametric",
            "custom_options": {},  # extra options to pass to your model
        }
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

# run the new command using the given tracer

# make a report, placing output in the current directory

# Serving and training loop
while True:
    print(pretty_print(dqn.train()))
    checkpoint_path = dqn.save()
    print("Last checkpoint", checkpoint_path)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(checkpoint_path)
