
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Union
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3 import PPO, DQN
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
import gymnasium as gym


class TrainTestManager:

    def __init__(self, train_env: Any, eval_env: Any, num_cpu: int = 4, policy_constraints: Dict[str, Dict[int, int]] = None) -> None:

        self.eval_env = eval_env
        self.env = train_env
        self.num_cpu = num_cpu

        self.policy_constraints = policy_constraints

        if not os.path.exists("./tensorboard"):
            os.makedirs("./tensorboard")
        if not os.path.exists("./logs"):
            os.makedirs("./logs")

        self.agents = self.env.possible_agents

        self.env = ss.pettingzoo_env_to_vec_env_v1(self.env)

        self.env = ss.concat_vec_envs_v1(
            self.env, num_vec_envs=self.num_cpu, num_cpus=self.num_cpu, base_class='stable_baselines3')

        self.eval_callback = EvalCallback(
            eval_env=self.env,
            best_model_save_path="./logs/",
            verbose=1, log_path="./logs/", eval_freq=200, deterministic=True, render=False)

        self.constrained_policy = copy.deepcopy(MlpPolicy)

        self.constrained_policy.predict = self.constrained_predict(
            copy.deepcopy(self.constrained_policy.predict), policy_constraints=policy_constraints)

    def constrained_predict(self, initial_predict: Callable, policy_constraints: Dict[str, Dict[int, int]] = None):
        """
        Create a constrained policy
        TODO: Generate a class inheriting from the Policy parent class -> a wrapper class
        """

        def predict(
            _self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

            actions, state = initial_predict(
                _self, observation, state, episode_start, deterministic)
            # print(actions)
            num_agent = len(self.agents)
            # observation_space = self.eval_env.venv.vec_envs[0].par_env.observation_space(
            #     agents[0])
            num_envs = self.num_cpu

            if policy_constraints is not None and observation.shape[0] == (num_agent * num_envs):

                for num_env in range(0, num_envs):
                    
                    env_obs = observation[(
                        num_env * num_agent):((num_env + 1)*num_agent)]

                    for agent_index, agent in enumerate(self.agents):

                        if agent in policy_constraints.keys():

                            agent_obs = str(env_obs[agent_index])

                            # print(agent_obs)
                            # print("="*30)

                            if agent_obs in policy_constraints[agent].keys():
                                actions[num_env * num_agent +
                                        agent_index] = policy_constraints[agent][agent_obs]

            # print(actions)
            return actions, state

        return predict

    def train(self):

        if not os.path.exists("./logs/best_model.zip"):

            print("Initiating training")

            # self.model = DQN(policy=MlpPolicy,
            #                  env=self.env, verbose=1, tensorboard_log="./tensorboard/")

            self.model = PPO(policy=self.constrained_policy,
                             env=self.env, verbose=1, tensorboard_log="./tensorboard/")

            # Almost infinite number of timesteps, but the training will converge at some point
            self.model.learn(total_timesteps=int(2e10),
                             callback=self.eval_callback, progress_bar=True)
            self.model.save("policy")

        else:
            print("Resuming training")
            self.model = PPO.load(path="./logs/best_model.zip", env=self.env,
                                  tensorboard_log="./tensorboard/")

            self.model.learn(total_timesteps=int(2e10),
                             callback=self.eval_callback, progress_bar=True)
            # self.model.save("policy")

    def test(self):

        print("Testing")

        self.eval_env = ss.pettingzoo_env_to_vec_env_v1(self.eval_env)

        self.eval_env = ss.concat_vec_envs_v1(
            self.eval_env, num_vec_envs=self.num_cpu, num_cpus=self.num_cpu, base_class='stable_baselines3')

        self.eval_env.reset()

        model = PPO.load(path="./logs/best_model.zip",
                         tensorboard_log="./tensorboard/")

        total_reward = 0
        frame_list = []

        NUM_RESETS = 1
        i = 0

        frame_list = [Image.fromarray(self.eval_env.render())]

        for i in range(NUM_RESETS):

            obs = self.eval_env.reset()
            
            for j in range(0, 22):

                action, _states = model.predict(obs)
                obs, rewards, dones, info = self.eval_env.step(action)

                total_reward += rewards[0]

                img = Image.fromarray(self.eval_env.render())
                frame_list.append(img)

                if dones[0]:
                    break

        # for i in range(NUM_RESETS):
        #     self.eval_env.reset(seed=42)
        #     for agent in self.eval_env.agent_iter():
        #         obs, rew, done, trunc, info = self.eval_env.last()

        #         act = 0
        #         # if str(obs) in self.policy_constraints[agent].keys():
        #         #     act = self.policy_constraints[agent][str(obs)]

        #         act = model.predict(obs, deterministic=True)[
        #             0] if not (done or trunc) else None
        #         # act = self.eval_env.action_space(
        #         #     agent).sample(mask=info["action_masks"])

        #         if trunc or done:
        #             act = None

        #         self.eval_env.step(act)

        #         i += 1
        #         if i % (len(self.eval_env.possible_agents)+1) == 0:
        #             total_reward = rew
        #             img = Image.fromarray(self.eval_env.render())
        #             frame_list.append(img)

        env_id = self.eval_env.venv.metadata["name"]
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
    eval_env = moving_company_v0.parallel_env(
        render_mode="rgb_array", size=10, seed=42)
    # eval_env = moving_company_v0.env(
    #     render_mode="rgb_array", size=10, seed=42, max_cycles=21)

    policy_constraints = {
        "agent_0": {
            "[0 1 0 0 2 0 0 1 0]": 1,
            "[0 5 0 0 2 0 0 1 0]": 5,
            "[0 4 0 0 3 0 0 1 0]": 2,
            "[0 1 0 0 3 0 0 1 0]": 2,
            "[0 1 0 0 3 0 0 4 1]": 6,
            "[0 1 0 0 3 0 0 4 2]": 6,
            "[0 1 0 0 2 0 0 5 1]": 0,
            "[0 1 0 0 2 0 0 5 2]": 0,
            "[0 1 0 0 2 0 0 4 3]": 0,
            "[0 1 0 0 2 0 0 4 1]": 0
        },
        "agent_1": {
            "[1 0 0 5 2 1 0 0 0]": 5,
            "[2 0 0 5 2 1 0 0 0]": 5,
            "[1 0 0 4 3 1 0 0 0]": 4,
            "[2 0 0 4 3 1 0 0 0]": 4,
            "[0 0 0 1 3 1 0 0 0]": 4,
            "[0 0 0 1 2 1 0 0 0]": 3,
            "[0 0 1 1 3 4 0 0 0]": 6,
            "[0 0 2 1 3 4 0 0 0]": 6,
            "[0 0 1 1 2 5 0 0 0]": 0,
            "[0 0 2 1 2 5 0 0 0]": 0,
            "[1 0 0 4 2 1 0 0 0]": 0,
            "[3 0 0 4 2 1 0 0 0]": 0,
            "[0 0 1 1 2 4 0 0 0]": 0,
            "[0 0 3 1 2 4 0 0 0]": 0

        },
        "agent_2": {
            "[0 1 0 0 2 0 1 5 0]": 5,
            "[0 1 0 0 2 0 2 5 0]": 5,
            "[0 1 0 0 3 0 1 4 0]": 1,
            "[0 1 0 0 3 0 2 4 0]": 1,
            "[0 4 0 0 2 0 0 1 0]": 2,
            "[0 1 0 0 2 0 0 1 0]": 2,
            "[0 1 0 0 3 0 0 1 0]": 1,
            "[0 4 0 0 3 0 0 1 0]": 6,
            "[0 5 0 0 2 0 0 1 0]": 0,
            "[0 1 0 0 2 0 1 4 0]": 0,
            "[0 1 0 0 2 0 3 4 0]": 0}
    }

    exenv = TrainTestManager(
        train_env=train_env,
        eval_env=eval_env,
        num_cpu=num_cpu)#,
        # policy_constraints=policy_constraints)

    if mode == "train":
        exenv.train()
    elif mode == "test":
        exenv.test()


if __name__ == "__main__":
    main()
    sys.exit(0)
