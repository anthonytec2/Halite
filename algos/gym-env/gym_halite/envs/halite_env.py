import gym
from gym import error, spaces, utils
from gym.utils import seeding
import hlt
import logging


class HaliteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game = hlt.Game()
        self.game.ready("MyPythonBot2")
        logging.info(
            "Successfully created bot! My Player ID is {}.".format(self.game.my_id))

    def step(self, action):
        self.game.end_turn(action)
        self.game.update_frame()
        return self.game

    def reset(self):
        sys.exit(0)

    def render(self, mode='human', close=False):
        sys.exit(0)
