from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from gym import spaces
import numpy as np
from six.moves import queue
import threading
import uuid
import ray
from ray.rllib.agents.dqn import DQNAgent
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.utils.policy_server import PolicyServer
from ray.rllib.models.model import Model
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.misc import normc_initializer, get_activation_fn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gym import spaces
from gym.spaces import Box, Discrete, Dict
import tensorflow.contrib.slim as slim
import tensorflow as tf
SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint.out"


class ParametricActionsModel(Model):
    """Parametric action model that handles the dot product and masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        action_mask = input_dict["obs"]["action_mask"]
        obs = input_dict["obs"]["real_obs"]
        # Standard FC net component.
        last_layer = obs
        hiddens = [256, 256]
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = slim.fully_connected(
                last_layer,
                size,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.tanh,
                scope=label)
        output = slim.fully_connected(
            last_layer,
            7,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="fc_out")
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        ouput_mask = output+inf_mask
        return ouput_mask, last_layer


class halite_env(ExternalEnv):
    def __init__(self):
        input_obs = Dict({
            "action_mask": spaces.Box(low=0, high=1, shape=(7, ), dtype=np.float32),
            "real_obs": spaces.Box(low=0, high=1, shape=(20, ), dtype=np.float32),
        })
        ExternalEnv.__init__(
            self, spaces.Discrete(7),
            input_obs)

    def run(self):
        print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                       SERVER_PORT))
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT)
        server.serve_forever()


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
    register_env("srv", lambda _: halite_env())

    # We use DQN since it supports off-policy actions, but you can choose and
    # configure any agent.
    dqn = DQNAgent(
        env="srv",
        config={
            "hiddens": [],  # don't postprocess the action scores
            # Use a single process to avoid needing to set up a load balancer
            "num_workers": 0,
            # Configure the agent to run short iterations for debugging
            "exploration_fraction": 0.01,
            "learning_starts": 100,
            "timesteps_per_iteration": 200,
            "model": {
                "custom_model": "pa_model",
                "custom_options": {},  # extra options to pass to your model
            }
        })

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        dqn.restore(checkpoint_path)

    # Serving and training loop
    while True:
        print(pretty_print(dqn.train()))
        checkpoint_path = dqn.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)
