from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import gym
import time
import subprocess
import sys

if __name__ == "__main__":
    while True:
        p = subprocess.Popen(
            '/Users/anthonybisulco/Documents/Cornell/Halite/run_game.sh', stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        # This makes the wait possible
        p_status = p.wait()
