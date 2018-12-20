from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import gym

from ray.rllib.utils.policy_client import PolicyClient
import hlt
from hlt import constants
from hlt.positionals import Direction
from hlt.positionals import Position
import random
import sys
import logging
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


def get_func(ship, action, me):
    if action == 0:
        return ship.move('n')
    elif action == 1:
        return ship.move('e')
    elif action == 2:
        return ship.move('s')
    elif action == 3:
        return ship.move('w')
    elif action == 4:
        return ship.make_dropoff()
    elif action == 5:
        return ship.stay_still()
    elif action == 6:
        return me.shipyard.spawn()


def format_observation(game, tf):
    board_distro = np.zeros((game.game_map.height, game.game_map.width))
    for i in range(0, game.game_map.height):
        for j in range(0, game.game_map.width):
            board_distro[i][j] = game.game_map[Position(i, j)].halite_amount
    bd = tf.transform(np.reshape(board_distro/1000, (1, 1024)))
    return bd


if __name__ == "__main__":
    client = PolicyClient("http://localhost:9900")
    game = hlt.Game()
    eid = client.start_episode(training_enabled=True)
    rewards = 0
    tf = torch.load(
        '/Users/anthonybisulco/Documents/Cornell/Halite/encoder.tf')
    game.ready("Ray-BOT")
    logging.info(
        "Successfully created bot! My Player ID is {}.".format(game.my_id))
    game.update_frame()
    game.end_turn([game.me.shipyard.spawn()])

    while True:
        game.update_frame()
        me = game.me
        game_map = game.game_map
        command_queue = []
        logging.info("START")

        obs = format_observation(game, tf)
        logging.info(obs.shape)
        action = client.get_action(eid, np.squeeze(obs))
        logging.info("GOT ACT")
        logging.info(action)
        for ship in me.get_ships():
            command_queue.append(get_func(ship, action, me))
        game.end_turn(command_queue)
        logging.info('UERE')
        logging.info(me.get_ships())
        for ship in me.get_ships():
            reward = ship.halite_amount
            logging.info('NOPE')
        client.log_returns(eid, reward)
        rewards += reward
        logging.info('DONE')
    logging.log("Total reward:", rewards)
    client.end_episode(eid, obs)
    exit(0)