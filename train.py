from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import gym
import time
import subprocess
import sys
import os
import signal
if __name__ == "__main__":
    while True:
        p = subprocess.Popen(
            os.path.join(os.getcwd(), 'run_game.sh'), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        (output, err) = p.communicate()
        # This makes the wait possible
        p_status = p.wait()
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except:
            print('Done')
