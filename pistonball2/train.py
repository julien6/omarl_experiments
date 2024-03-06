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

    print("Training")

    if not os.path.exists("./tensorboard"):
        os.makedirs("./tensorboard")
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True,
                                     random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    env.reset(42)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 8, num_cpus=4, base_class='stable_baselines3')

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=1500, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path="./logs/", callback_on_new_best=callback_on_best,
                                 verbose=1, log_path="./logs/", eval_freq=500, deterministic=True, render=False)

    model = PPO(CnnPolicy, env, verbose=1, tensorboard_log="./tensorboard/", gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
                vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)

    # if not os.path.exists("./sb3_logs"):
    #     os.makedirs("./sb3_logs")
    # tmp_path = "./sb3_logs"
    # # set up logger
    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # model.set_logger(new_logger)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    if not os.path.exists("./policy.zip"):

        # Almost infinite number of timesteps, but the training will stop
        # early as soon as the reward threshold is reached
        model.learn(total_timesteps=int(1e10),
                    callback=[checkpoint_callback, eval_callback], progress_bar=True)
        model.save("policy")

    else:
        print("Resuming training")
        model = PPO.load(path="policy.zip", env=env,
                         tensorboard_log="./tensorboard/")
        model.learn(total_timesteps=int(1e10),
                    callback=[checkpoint_callback, eval_callback], progress_bar=True)
        model.save("policy")

    print("Rendering")
    env = pistonball_v6.env(render_mode="rgb_array", n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True,
                            random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    env.reset(42)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    model = PPO.load("policy", tensorboard_log="./tensorboard/")
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
