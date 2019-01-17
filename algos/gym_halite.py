import os
import socket

import gym
import numpy as np
import psutil
from gym import spaces
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
        self.conn.sendall(act_msg.encode("utf-8"))
        data = self.conn.recv(272)
        res = np.frombuffer(data, dtype=np.float32)
        obs = res[:27]
        mask = res[27:34]
        reward = res[-2]
        done = res[-1]
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
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cmd = f"{os.path.join(dir_path,'halite')} --replay-directory {os.path.join(dir_path,'replays')} -vvv --width 32 --height 32 --no-timeout --no-logs 'python3.6 {os.path.join(dir_path,'bots','networking.py')} --port={self.server_address}' 'python3.6 {os.path.join(dir_path,'bots','Bot2.py')}' &"
        self.res = psutil.Popen(cmd, shell=True)
        self.s.listen(1)
        self.conn, _ = self.s.accept()
