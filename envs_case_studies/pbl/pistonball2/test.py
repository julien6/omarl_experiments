from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import os
from array2gif import write_gif
import numpy as np
import matplotlib.pyplot as plt
import sys
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback


def main():

    print("Rendering")
    env = pistonball_v6.env(render_mode="rgb_array", n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True,
                            random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    env.reset(42)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    model = PPO.load(path="./logs/best_model.zip", tensorboard_log="./tensorboard/")
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
            act = model.predict(obs, deterministic=True)[
                0] if not done else None
            env.step(act)
            trajectories[agent].append((obs, act, rew))
            total_reward += rew
            i += 1
            if i % (len(env.possible_agents)+1) == 0:
                obs_list.append(np.transpose(
                    env.render(), axes=(1, 0, 2)))

    env.close()

    # print({agent: [int(act[0]) if act is not None else 0 for _,
    #                act, _ in data] for agent, data in trajectories.items()})
    print("average total reward: ", total_reward/NUM_RESETS)
    write_gif(obs_list, 'pistonball_test.gif', fps=15)

    # # Plotting
    # rewards = model.logger.
    # plt.plot(rewards)
    # plt.title('Évolution de la Récompense en fonction des Itérations')
    # plt.xlabel('Itérations')
    # plt.ylabel('Récompense')
    # plt.savefig('rewards.png')
    # # plt.show()

    print("Finished")


if __name__ == "__main__":
    main()
    sys.exit(0)
