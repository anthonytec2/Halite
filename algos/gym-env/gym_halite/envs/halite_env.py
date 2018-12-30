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
HOST = '127.0.0.1'
PORT = 0

import time


class HaliteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def step(self, action):
        act_msg = 'A '+str(action)
        self.conn.send(act_msg.encode("utf-8"))
        data = self.conn.recv(4096)
        res = np.frombuffer(data)
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
        data = self.conn.recv(4096)
        return np.frombuffer(data)

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
        print(f'Bound Localhost Port: {self.s.getsockname()[1]}')
        cmd = f'./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3 bots/networking.py --port={self.s.getsockname()[1]}" "python3 bots/Bot2.py" --no-timeout'
        self.res = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        self.s.listen()
        self.conn, self.addr = self.s.accept()


if __name__ == '__main__':
    new_env = HaliteEnv()
    new_env.reset()
    new_env.step(1)
    time.sleep(0.3)
    new_env.step(1)
    time.sleep(0.3)
    new_env.step(1)
    time.sleep(0.3)
    new_env.s.close()
