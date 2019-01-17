import argparse
import logging
import os
import socket
import sys

import numpy as np
import zmq
from sklearn.decomposition import TruncatedSVD

import hlt
from hlt import constants
from hlt.positionals import Position
import pickle
HOST = '127.0.0.1'


def cleanup():
    for ship in me.get_ships():
        halite_ship = ship.halite_amount
    obs = get_obs(game, tf)
    valid_mask = is_valid_move(game)
    new_ar = np.hstack((obs, valid_mask))
    send_obs = np.append(new_ar, [0, 1]).astype(np.float32)
    s.sendall(send_obs.tostring())
    game.end_turn(command_queue)
    s.close()
    sys.exit(1)


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

    for ship in me.get_ships():
        hal_cell = game.game_map[ship.position].halite_amount
        if ship.halite_amount < hal_cell:
            action_mask[0:4] = 0
    return action_mask


# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--port", help="Ports IP", default="", type=str)
args = parser.parse_args()

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(args.port)
    # Start Up Game
    game = hlt.Game()
    game.ready("Ray-BOT-Networking-V4")
    game.update_frame()

    # Spawn Ship and Start new turn
    game.end_turn([game.me.shipyard.spawn()])
    game.update_frame()

    # Send first observation to reset
    for ship in game.me.get_ships():
        halite_ship = ship.halite_amount
    reward_base = game.me.halite_amount+.01*halite_ship
    with open('/home/abisulco/Halite/bots/encoder.pkl', 'rb') as f:
        tf = pickle.load(f)
    obs = get_obs(game, tf)
    res = np.ones(7)
    res[6] = 0
    res[5] = 0
    new_ar = np.hstack((obs, res)).astype(np.float32)
    s.sendall(new_ar.tostring())
    i = 0
    while True:
        i += 1
        # RX Action and Peform
        data = s.recv(1).decode()
        action = int(data)
        # Update Constants and get next actions
        me = game.me
        game_map = game.game_map
        command_queue = []
        for ship in me.get_ships():
            command_queue.append(get_func(ship, action, me))
        # Check if last turn if so run cleanup
        if game.turn_number == constants.MAX_TURNS:
            cleanup()
        # End the turn and update commands
        game.end_turn(command_queue)

        game.update_frame()

        # Send next valid observations
        for ship in me.get_ships():
            halite_ship = ship.halite_amount
        reward = (game.me.halite_amount+.01*halite_ship)-reward_base
        reward_base = reward
        obs = get_obs(game, tf)
        valid_mask = is_valid_move(game)
        new_ar = np.hstack((obs, valid_mask))
        send_obs = np.append(new_ar, [reward, 0]).astype(np.float32)
        s.sendall(send_obs.tostring())
