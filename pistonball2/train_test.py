from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import os
from array2gif import write_gif
import numpy as np


if not os.path.exists("./policy.zip"):
    env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True,
                                     random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 8, num_cpus=4, base_class='stable_baselines3')
    model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
                vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
    model.learn(total_timesteps=2000000)
    model.save("policy")

# Rendering
env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

model = PPO.load("policy")
obs_list = []

total_reward = 0
obs_list = []
NUM_RESETS = 1
i = 0

env.reset()
trajectories = {agent: [] for agent in env.agents}

for i in range(NUM_RESETS):
    env.reset()
    for agent in env.agent_iter():
        obs, rew, done, _, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        trajectories[agent].append((obs, act, rew))
        total_reward += rew
        i += 1
        if i % (len(env.possible_agents)+1) == 0:
            obs_list.append(np.transpose(
                env.render(), axes=(1, 0, 2)))

env.close()
print({agent: [int(act[0]) if act is not None else 0 for _,
               act, _ in data] for agent, data in trajectories.items()})
print("average total reward: ", total_reward/NUM_RESETS)
write_gif(obs_list, 'pistonball_test.gif', fps=15)
