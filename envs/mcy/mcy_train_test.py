
import copy
from typing import Any
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3 import PPO
import supersuit as ss
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image
from MovingCompany.movingcompany import moving_company_v0


class TrainTestManager:

    def __init__(self, train_env: Any, eval_env: Any, num_cpu: int = 4) -> None:

        self.eval_env = eval_env
        self.env = train_env
        self.num_cpu = num_cpu

        if not os.path.exists("./tensorboard"):
            os.makedirs("./tensorboard")
        if not os.path.exists("./logs"):
            os.makedirs("./logs")

        self.env = ss.pettingzoo_env_to_vec_env_v1(self.env)

        self.env = ss.concat_vec_envs_v1(
            self.env, num_vec_envs=self.num_cpu, num_cpus=self.num_cpu, base_class='stable_baselines3')

        self.eval_callback = EvalCallback(
            eval_env=self.env,
            best_model_save_path="./logs/",
            verbose=1, log_path="./logs/", eval_freq=200, deterministic=True, render=False)

    def train(self):

        if not os.path.exists("./logs/best_model.zip"):

            print("Initiating training")

            self.model = PPO(policy=MlpPolicy,
                             env=self.env, verbose=1, tensorboard_log="./tensorboard/", gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
                             vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)

            # Almost infinite number of timesteps, but the training will converge at some point
            self.model.learn(total_timesteps=int(2e10),
                             callback=self.eval_callback, progress_bar=True)
            # self.model.save("policy")

        else:
            print("Resuming training")
            self.model = PPO.load(path="./logs/best_model.zip",
                                  tensorboard_log="./tensorboard/")

            self.model.learn(total_timesteps=int(2e10),
                             callback=self.eval_callback, progress_bar=True)
            # self.model.save("policy")

    def test(self, manual_policy=False):

        print("Testing")

        self.eval_env.reset(seed=42)

        model = PPO.load(path="./logs/best_model.zip",
                            tensorboard_log="./tensorboard/")

        total_reward = 0
        frame_list = []

        NUM_RESETS = 1
        i = 0

        perfect_policy = [5, 0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 6, 0, 0, 0, 5, 0, 0, 4, 0, 0, 4,
                          0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 6, 0, 0, 0, 5, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 6]

        frame_list = [Image.fromarray(self.eval_env.render())]

        for i in range(NUM_RESETS):
            self.eval_env.reset(seed=42)
            for agent in self.eval_env.agent_iter():
                obs, rew, done, _, info = self.eval_env.last()

                act = model.predict(obs, deterministic=True)[
                    0] if not done else None

                self.eval_env.step(act)

                i += 1
                if i % (len(self.eval_env.possible_agents)+1) == 0:
                    total_reward = rew
                    img = Image.fromarray(self.eval_env.render())
                    frame_list.append(img)

        env_id = self.eval_env.metadata["name"]
        frame_list[0].save(f"{env_id}.gif", save_all=True,
                           append_images=frame_list[1:], duration=10, loop=0)

        self.eval_env.close()

        print("Average reward: ", total_reward / NUM_RESETS)

        print("Finished")


def main():
    args = sys.argv[1:]

    if not ((len(args) == 4 and args[0] == "-mode" and args[2] == "-num_cpu" and args[1] in ["train", "test"] and args[3].isdigit()) or
            (len(args) == 2 and args[0] == "-mode" and args[1] in ["train", "test"])):
        print("Usage: python mcy_train_test.py -mode <train/test> -num_cpu <num_cpu>")
        sys.exit()

    mode = args[1]
    num_cpu = 4

    if len(args) == 4:
        num_cpu = int(args[3])

    train_env = moving_company_v0.parallel_env(
        render_mode="grid", size=10, seed=42)
    eval_env = moving_company_v0.env(render_mode="rgb_array", size=10, seed=42)

    exenv = TrainTestManager(
        train_env=train_env,
        eval_env=eval_env,
        num_cpu=num_cpu)

    if mode == "train":
        exenv.train()
    elif mode == "test":
        exenv.test(manual_policy=False)


if __name__ == "__main__":
    main()
    sys.exit(0)
