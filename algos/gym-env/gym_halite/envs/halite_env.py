import gym
from gym import error, spaces, utils
from gym.utils import seeding
import hlt
import logging
import subprocess
import psutil
import socket
import sys
import numpy as np
from gym.spaces import Box, Discrete, Dict

import time

HOST = '127.0.0.1'
PORT = 0

class HaliteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
         input_obs = Dict({
            "action_mask": spaces.Box(low=0, high=1, shape=(7, ), dtype=np.float32),
            "real_obs": spaces.Box(low=0, high=1, shape=(27, ), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(7)
        self.observation_space = input_obs

    def step(self, action):
        act_msg = 'A '+str(action)
        self.conn.send(act_msg.encode("utf-8"))
        data = self.conn.recv(249)
        res = np.frombuffer(data, dtype=np.float32)
        obs = res[:27]
        mask = res[27:34]
        reward = res[-2]
        done = res[-1]
        done = True if done == 1 else False
        obs = {
            "action_mask": mask,
            "real_obs": obs,
        }
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        self.establish_conn()
        data = self.conn.recv(249)
        return np.frombuffer(data, dtype=np.float32)

    def establish_conn(self):
        try:
            self.s.close()
            del self.s, self.conn, self.addr
        except:
            o = 2
        try:
            self.res.kill()
        except:
            o = 1

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        # print(f'Bound Localhost Port: {self.s.getsockname()[1]}')
        cmd = f'./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3 bots/networking.py --port={self.s.getsockname()[1]}" "python3 bots/Bot2.py" --no-timeout'
        self.res = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        self.s.listen()
        self.conn, self.addr = self.s.accept()
        # print('ESTABLISHED CONNECTION HALITE_ENV.PY')


if __name__ == '__main__':
    new_env = HaliteEnv()
    new_env.reset()
    done = False
    while True:
        obs, reward, done, _ = new_env.step(4)
        if done:
            new_env.s.close()
            sys.exit(1)
