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
import signal
import sys
import logging
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
import traceback
import pdb
import os


def signal_handler(sig, frame):
    logging.info('You pressed Ctrl+C!')
    '''
    for ship in me.get_ships():
        halite_ship = ship.halite_amount
        reward = (game.me.halite_amount+.01*halite_ship)-reward_base
        reward_base = reward
    client.log_returns(eid, reward)
    '''


def cleanup():
    client.log_returns(eid, 0)
    client.end_episode(eid, obs)
    game.end_turn(command_queue)
    sys.exit()


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
        return ship.stay_still()
    elif action == 5:
        return ship.make_dropoff()
    elif action == 6:
        return me.shipyard.spawn()


def get_obs(game, tf):
    board_distro = np.zeros((game.game_map.height, game.game_map.width))
    for i in range(0, game.game_map.height):
        for j in range(0, game.game_map.width):
            board_distro[i][j] = game.game_map[Position(i, j)].halite_amount
    bd = tf.transform(np.reshape(board_distro/1000, (1, 1024)))
    logging.info(game.me.get_ships())
    for ship in game.me.get_ships():
        posagentX = ship.position.x
        posagentY = ship.position.y
        shipHal = ship.halite_amount/1000
    posdpX = game.me.shipyard.position.x
    posdpY = game.me.shipyard.position.y
    halite = game.me.halite_amount/1e5
    turn_num = game.turn_number/constants.MAX_TURNS
    return np.append(bd, [posagentX, posagentY, shipHal, posdpX, posdpY, halite, turn_num])


def is_valid_move(game):
    action_mask = np.ones(7)
    # if game.me.halite_amount < 1000:
    action_mask[6] = 0
    # if game.me.halite_amount < 4000:
    action_mask[5] = 0

    logging.info(me.get_ships())
    for ship in me.get_ships():
        hal_cell = game.game_map[ship.position].halite_amount
        logging.info('Halite Cell: '+str(hal_cell) +
                     ' Halite Ship: ' + str(ship.halite_amount))
        if ship.halite_amount < hal_cell:
            action_mask[0:4] = 0
    logging.info(action_mask)
    return action_mask


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    client = PolicyClient("http://localhost:9900")
    game = hlt.Game()
    eid = client.start_episode(training_enabled=True)
    rewards = 0
    tf = torch.load(os.path.join(os.getcwd(), 'encoder.tf'))
    game.ready("Ray-BOT")
    logging.info(
        "Successfully created bot! My Player ID is {}.".format(game.my_id))
    game.update_frame()
    game.end_turn([game.me.shipyard.spawn()])
    reward_base = game.me.halite_amount
    while True:
        game.update_frame()
        me = game.me
        game_map = game.game_map
        command_queue = []
        logging.info('IDK')
        obs = get_obs(game, tf)
        valid_mask = is_valid_move(game)
        logging.info(obs)
        obs = {
            "action_mask": valid_mask,
            "real_obs": obs,
        }

        action = client.get_action(eid, obs)
        logging.info(action)
        logging.info(game.me.halite_amount)
        for ship in me.get_ships():
            command_queue.append(get_func(ship, action, me))
        if game.turn_number == constants.MAX_TURNS:
            cleanup()
        game.end_turn(command_queue)
        for ship in me.get_ships():
            halite_ship = ship.halite_amount
        reward = (game.me.halite_amount+.01*halite_ship)-reward_base
        reward_base = reward
        client.log_returns(eid, reward)
        rewards += reward
