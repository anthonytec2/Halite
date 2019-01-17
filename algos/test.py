from __future__ import absolute_import, division, print_function
import socket
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
import time
CHECKPOINT_FILE = "last_checkpoint.out"
HOST = '127.0.0.1'
PORT = 0
logger = logging.getLogger('lapp')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('lamp.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


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
        logger.debug(f'GYM Action: {action} {len(act_msg.encode("utf-8"))}')
        self.conn.sendall(act_msg.encode("utf-8"))
        data = self.conn.recv(272)
        res = np.frombuffer(data, dtype=np.float32)
        logger.debug(f'GYM OBS: {res}')
        obs = res[:27]
        mask = res[27:34]
        reward = res[-2]
        done = res[-1]
        logger.debug(f'GYM DONE: {done}')
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
        logger.debug(f'GYM OBS: {res}')
        obs = res[:27]
        mask = res[27:34]
        obs_ret = {
            "action_mask": mask,
            "real_obs": obs,
        }
        return obs_ret

    def establish_conn(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))

    def run_program(self):
        logger.debug('RUN -----------------------')
        cmd = './halite --replay-directory replays/ -vvv --width 32 --height 32 --no-timeout --no-logs --no-replay "python3.6 bots/networking.py --port={}" "python3.6 bots/Bot2.py" &'.format(
            self.s.getsockname()[1])
        self.res = psutil.Popen(cmd, shell=True)
        self.s.listen(1)
        self.conn, addr = self.s.accept()


i = 12
env_ls = []
for d in range(i):
    env_ls.append(HaliteEnv({}))
i = 0
while True:
    for env in env_ls:
        env.reset()
    while True:
        for env in env_ls:
            obs_ret, reward, done, _ = env.step(4)
        i += 1
        if done:
            print('hit')
            break
