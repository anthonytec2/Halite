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
from ray.rllib.agents.dqn import DQNAgent
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.misc import get_activation_fn, normc_initializer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import logging
import socket
CHECKPOINT_FILE = "last_checkpoint.out"
HOST = '127.0.0.1'
PORT = 0


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
        #logger.debug(f'GYM Action: {action} {len(act_msg.encode("utf-8"))}')
        self.conn.sendall(act_msg.encode("utf-8"))
        data = self.conn.recv(272)
        res = np.frombuffer(data, dtype=np.float32)
        #logger.debug(f'GYM OBS: {res}')
        obs = res[:27]
        mask = res[27:34]
        reward = res[-2]
        done = res[-1]
        #logger.debug(f'GYM DONE: {done}')
        done = True if done == 1 else False
        obs_ret = {
            "action_mask": mask,
            "real_obs": obs,
        }

        return obs_ret, reward, done, {}

    def reset(self):
        self.run_program()
        data = self.conn.recv(136)
        res = np.frombuffer(data, dtype=np.float32)
        # .debug(f'GYM OBS: {res}')
        obs = res[:27]
        mask = res[27:34]
        obs_ret = {
            "action_mask": mask,
            "real_obs": obs,
        }
        return obs_ret

    def establish_conn(self):
        self.server_address = f'/tmp/v-{np.random.random()}'
        try:
            os.unlink(self.server_address)
        except OSError:
            if os.path.exists(self.server_address):
                raise
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.bind(self.server_address)

    def run_program(self):
        #logger.debug('RUN -----------------------')
        cmd = './halite --replay-directory replays/ -vvv --width 32 --height 32 --no-timeout --no-logs --no-replay "python3.6 bots/networking.py --port={}" "python3.6 bots/Bot2.py" &'.format(
            self.server_address)
        self.res = psutil.Popen(cmd, shell=True)
        self.s.listen(1)
        self.conn, addr = self.s.accept()


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


ray.init()
ModelCatalog.register_custom_model("parametric", ParametricActionsModel)
register_env("halite_env", env_creator)
dqn = DQNAgent(
    env="halite_env",
    config={
        "env_config": {},
        # Use a single process to avoid needing to set up a load balancer
        "num_workers": 6,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 4,
        "num_gpus": 0,
        "hiddens": [],
        "schedule_max_timesteps": 100000000,
        # Number of env steps to optimize for before returning
        "timesteps_per_iteration": 1000,
        # Fraction of entire training period over which the exploration rate is
        # annealed
        "exploration_fraction": 0.8,
        # Final value of random action probability
        "exploration_final_eps": 0.02,
        # Update the target network every `target_network_update_freq` steps.serrv
        "target_network_update_freq": 500,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 50000,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Fraction of entire training period over which the beta parameter is
        # annealed
        "beta_annealing_fraction": 0.2,
        # Final value of beta
        "final_prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # Whether to LZ4 compress observations
        "compress_observations": True,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 1e-3,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_norm_clipping": 40,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "sample_batch_size": 4,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 32,
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
