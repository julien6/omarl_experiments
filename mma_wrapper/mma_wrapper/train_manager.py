
import copy
import random
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
from custom_envs.movingcompany import moving_company_v0

from mma_wrapper.policy_model import joint_policy_constraint

# from mma_wrapper import probabilistic_decision_graph


# from custom_envs.movingcompany import moving_company_v0
# from mma_wrapper import mma_wrapper
# env = moving_company_v0.parallel_env(render_mode="human")

# hist_to_specs = lambda hist: {"role": "horizontal_mover"} if "3" in [
#     act for obs, act in hist.items()] else None
# agt_to_cons_specs = {"agent_0": {"[0 5 0 0 2 0 0 1 0]": 5}}
# env = mma_wrapper(env, hist_to_specs, agt_to_cons_specs, "CORRECT", ["sequence_clustering"], ["role", "plan"], ["dendogram", "PCA"])
# env.train("PPO_default")
# raw_specs, agent_to_specs = env.generate_specs()


class train_test_manager:

    def __init__(self, train_env: Any, eval_env: Any, label_to_obj: Dict[str, Any], num_cpu: int = 4, policy_constraints: joint_policy_constraint = None, mode=0) -> None:

        self.eval_env = eval_env
        self.env = train_env
        self.num_cpu = num_cpu

        self.label_to_obj: Dict[str, Any] = label_to_obj
        self.obj_to_label: Dict[Any, str] = {
            str(v): k for k, v in self.label_to_obj.items()}

        self.policy_constraints = policy_constraints

        self.agents = self.env.aec_env.possible_agents

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

    def constrained_predict(self, initial_predict: Callable, policy_constraints: joint_policy_constraint = None):
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

                    joint_observations = {agent: self.obj_to_label.get(str(
                        env_obs[agent_index]), str(env_obs[agent_index])) for agent_index, agent in enumerate(self.agents)}

                    next_joint_actions = policy_constraints\
                        .next_actions(joint_observations)

                    next_joint_actions = {agent: action_labels if len(action_labels) > 0 else [
                        self.obj_to_label["0"]] for agent, action_labels in next_joint_actions.items()}

                    next_joint_actions = {agent: random.choice([self.label_to_obj[action_label]
                                                                for action_label in action_labels]) for agent, action_labels in next_joint_actions.items()}
                    for agent_index, agent in enumerate(self.agents):
                        actions[num_env * num_agent +
                                agent_index] = next_joint_actions[agent]

            # print(actions)
            return actions, state

        return predict

    def train(self, total_step=int(2e10)):

        if not os.path.exists("./logs/best_model.zip"):

            print("Initiating training")

            # self.model = DQN(policy=MlpPolicy,
            #                  env=self.env, verbose=1, tensorboard_log="./tensorboard/")

            self.model = PPO(policy=self.constrained_policy,
                             env=self.env, verbose=1, tensorboard_log="./tensorboard/")

            # Almost infinite number of timesteps, but the training will converge at some point
            self.model.learn(total_timesteps=total_step,
                             callback=self.eval_callback, progress_bar=True)
            # self.model.save("policy")

        else:

            print("Resuming training")
            self.model = PPO.load(path="./logs/best_model.zip", env=self.env,
                                  tensorboard_log="./tensorboard/")

            self.model.learn(total_timesteps=total_step,
                             callback=self.eval_callback, progress_bar=True)
            # self.model.save("policy")

    def test(self):

        print("Testing")

        self.eval_env = ss.pettingzoo_env_to_vec_env_v1(self.eval_env)

        # A single env to generate histories
        self.eval_env = ss.concat_vec_envs_v1(
            self.eval_env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

        def concatenate_obs(obs: np.ndarray):
            """Used to adapt the observation/actions to the n concatenated environments model
            """
            return np.tile(obs, (self.num_cpu, 1))

        def deconcatenate_act(act: np.ndarray):
            """Used to adapt the observation/actions to the n concatenated environments model
            """
            return act[:int(act.shape[0] / self.num_cpu)]

        def add_to_history(joint_history, joint_observation: np.array, joint_action: np.array):
            """Used to split the vectorized obs and actions into the same agents' vectors
            """
            for agent_index, history in enumerate(joint_history):
                agent_history = copy.copy(history)
                agent_history += [(joint_observation[agent_index],
                                   joint_action[agent_index])]
                joint_history[agent_index] = agent_history
            return joint_history

        self.eval_env.reset()

        model = PPO.load(path="./logs/best_model.zip",
                         tensorboard_log="./tensorboard/")

        total_reward = 0
        frame_list = []
        num_step = 22

        joint_histories = []

        num_it = 1
        num_ep = 1

        frame_list = [Image.fromarray(self.eval_env.render())]

        for it in range(num_it):

            for ep in range(num_ep):

                joint_history = [[]] * len(self.agents)

                obs = self.eval_env.reset()

                for j in range(0, num_step):

                    action, _states = model.predict(concatenate_obs(obs))
                    action = deconcatenate_act(action)

                    joint_history = add_to_history(joint_history, obs, action)

                    obs, rewards, dones, info = self.eval_env.step(action)

                    total_reward += rewards[0]

                    img = Image.fromarray(self.eval_env.render())
                    frame_list.append(img)

                    if dones[0]:
                        break

                joint_histories += [joint_history]

        env_id = self.eval_env.venv.metadata["name"]
        frame_list[0].save(f"{env_id}.gif", save_all=True,
                           append_images=frame_list[1:], duration=10, loop=0)

        self.eval_env.close()

        print("Average reward: ", total_reward / num_it)
        print("Finished")

        return joint_histories


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

    exenv = train_test_manager(
        train_env=train_env,
        eval_env=eval_env,
        num_cpu=num_cpu,
        policy_constraints=policy_constraints)

    if mode == "train":
        exenv.train()
    elif mode == "test":
        exenv.test()


if __name__ == "__main__":
    main()
    sys.exit(0)
