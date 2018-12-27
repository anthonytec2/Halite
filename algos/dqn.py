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
from ray.rllib.agents.ppo import PPOAgent
import trace
import sys
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
SERVER_ADDRESS = "0.0.0.0"
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
        hiddens = [20, 20, 15, 15, 10, 10, 9, 9]
        # hiddens = [256, 256]
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
            "real_obs": spaces.Box(low=0, high=1, shape=(27, ), dtype=np.float32),
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
            # Use a single process to avoid needing to set up a load balancer
            "num_workers": 0,
            "hiddens": [],
            # === Exploration ===
            # Max num timesteps for annealing schedules. Exploration is annealed from
            # 1.0 to exploration_fraction over this number of timesteps scaled by
            # exploration_fraction
            "schedule_max_timesteps": 10000000,
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 1000,
            # Fraction of entire training period over which the exploration rate is
            # annealed
            "exploration_fraction": 0.8,
            # Final value of random action probability
            "exploration_final_eps": 0.02,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 500,

            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": 50000,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            # Alpha parameter for prioritized replay buffer.
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Fraction of entire training period over which the beta parameter is
            # annealed
            "beta_annealing_fraction": 0.2,
            # Final value of beta
            "final_prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # Whether to LZ4 compress observations
            "compress_observations": True,

            # === Optimization ===
            # Learning rate for adam optimizer
            "lr": 1e-3,
            # Adam epsilon hyper parameter
            "adam_epsilon": 1e-8,
            # If not None, clip gradients during optimization at this value
            "grad_norm_clipping": 40,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1000,
            # Update the replay buffer with this many samples at once. Note that
            # this setting applies per-worker if num_workers > 1.
            "sample_batch_size": 4,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 32,

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

    # run the new command using the given tracer

    # make a report, placing output in the current directory

    # Serving and training loop
    while True:

        print(pretty_print(dqn.train()))
        checkpoint_path = dqn.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)
