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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_jobs", help="Number of Jobs", default=16,
                        type=str)
    parser.add_argument("--ip", help="Number of Jobs", default="35.243.173.101",
                        type=str)
    args = parser.parse_args()
    while True:
        p = subprocess.Popen(
            os.path.join(os.getcwd(), f'run_game.sh {args.num_jobs} {args.ip}'), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        (output, err) = p.communicate()
        # This makes the wait possible
        p_status = p.wait()
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except:
            print('Done')
