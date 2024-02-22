# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !sudo apt-get update
# !sudo apt-get install -y xvfb ffmpeg freeglut3-dev
# !pip install 'imageio==2.4.0'
# !pip install pyvirtualdisplay
# !pip install tf-agents[reverb]
# !pip install pyglet xvfbwrapper
# !pip install tf-keras

from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.policies import py_tf_eager_policy
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.drivers import py_driver
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.agents.ppo import ppo_agent
import tensorflow as tf
import reverb
import pyvirtualdisplay
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import IPython
import imageio
import base64

from sim_v2.kast_tf_env import Kast

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'


# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

num_iterations = 100
collect_episodes_per_iteration = 2
replay_buffer_capacity = 2000

fc_layer_params = (100,)

learning_rate = 1e-3
log_interval = 25
num_eval_episodes = 1
eval_interval = 50

# train_py_env = suite_gym.load(env_name)
# eval_py_env = suite_gym.load(env_name)
train_py_env = Kast()
eval_py_env = Kast()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

value_net = value_network.ValueNetwork(
    train_env.observation_spec(),
    fc_layer_params=fc_layer_params,
    activation_fn=tf.keras.activations.relu
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

tf_agent = ppo_agent.PPOAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=optimizer,
    train_step_counter=train_step_counter)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)


def collect_episode(environment, policy, num_episodes):

    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
        [rb_observer],
        max_episodes=num_episodes)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


plt.plot(range(0, len(returns)), returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig('kast-ppo-learning-curve.png')
plt.close()


######### Evaluation #########

# num_episodes = 3
# video_filename = 'imageio.mp4'
# with imageio.get_writer(video_filename, fps=60) as video:
#     for _ in range(num_episodes):
#         time_step = eval_env.reset()
#         video.append_data(eval_py_env.render())
#         while not time_step.is_last():
#             action_step = tf_agent.policy.action(time_step)
#             time_step = eval_env.step(action_step.action)
#             video.append_data(eval_py_env.render())

eval_num_episodes = 1
returns = []
for _ in range(eval_num_episodes):
    time_step = eval_env.reset()
    returns.append(eval_py_env.render())
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        print(action_step.action)
        time_step = eval_env.step(action_step.action)
        returns.append(eval_py_env.render())

plt.plot(range(0, len(returns)), returns)
plt.ylabel('Value over an episode')
plt.xlabel('Steps')
plt.savefig('kast-ppo-rendered.png')
plt.close()
