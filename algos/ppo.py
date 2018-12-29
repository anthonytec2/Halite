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
SERVER_PORT = 9902
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

    # We use PPO since it supports off-policy actions, but you can choose any
    # configure any agent.
    ppo = PPOAgent(
        env="srv",
        config={
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": True,
            # GAE(lambda) parameter
            "lambda": 1.0,
            # Initial coefficient for KL divergence
            "kl_coeff": 0.2,
            # Size of batches collected from each worker
            "sample_batch_size": 200,
            # Number of timesteps collected for each SGD round
            "train_batch_size": 4000,
            # Total SGD batch size across all devices for SGD
            "sgd_minibatch_size": 128,
            # Number of SGD iterations in each outer loop
            "num_sgd_iter": 30,
            # Stepsize of SGD
            "lr": 5e-3,
            # Learning rate schedule
            "lr_schedule": None,
            # Share layers for value function
            "vf_share_layers": True,
            # Coefficient of the value function loss
            "vf_loss_coeff": 1.0,
            # Coefficient of the entropy regularizer
            "entropy_coeff": 0.0,
            # PPO clip parameter
            "clip_param": 0.3,
            # Clip param for the value function. Note that this is sensitive to the
            # scale of the rewards. If your expected V is large, increase this.
            "vf_clip_param": 10.0,
            # Target value for KL divergence
            "kl_target": 0.01,
            # Whether to rollout "complete_episodes" or "truncate_episodes"
            "batch_mode": "truncate_episodes",
            # Which observation filter to apply to the observation
            "observation_filter": "NoFilter",
            # Uses the sync samples optimizer instead of the multi-gpu one. This does
            # not support minibatches.
            "simple_optimizer": False,
            # (Deprecated) Use the sampling behavior as of 0.6, which launches extra
            # sampling tasks for performance but can waste a large portion of samples.
            "straggler_mitigation": False,

            "model": {
                "custom_model": "pa_model",
                "custom_options": {},  # extra options to pass to your model
            }
        })

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        ppo.restore(checkpoint_path)

    # run the new command using the given tracer

    # make a report, placing output in the current directory

    # Serving and training loop
    while True:

        print(pretty_print(ppo.train()))
        checkpoint_path = ppo.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)
