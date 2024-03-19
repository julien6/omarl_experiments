from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import math
from pprint import pprint


import os

import abc
from typing import Dict, List
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.policies import random_tf_policy

import tensorflow as tf
import random

os.environ['TF_USE_LEGACY_KERAS'] = '1'


class Kast(py_environment.PyEnvironment):

    def __init__(self, seed=42):

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=10, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._step_number = 0
        self._reward = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._step_number = 0
        self._reward = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def render(self):
        print(self._state)
        return self._state

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        if 0 <= action and action <= 10:
            self._state = action
        else:
            raise ValueError('`action` should be 0,..,9')

        # self._reward += 10 if self._state == 1 else -10

        if (self._step_number == 5):
            self._state = 7

        if (self._step_number == 9):
            self._state = 2

        if self._step_number > 20:
            self._episode_ended = True
            self._reward = 100 if self._state == 1 else -100
            return ts.termination(np.array([self._state], dtype=np.int32), self._reward)
        else:
            self._step_number += 1
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0, discount=1.0)


class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
