from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import gym

from ray.rllib.utils.policy_client import PolicyClient


if __name__ == "__main__":
    client = PolicyClient("http://localhost:9900")

    eid = client.start_episode(training_enabled=True)

    rewards = 0

    while True:
        action = client.get_action(eid, obs)
        obs, reward, done, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward)
        if done:
            print("Total reward:", rewards)
            client.end_episode(eid, obs)
            exit(0)
