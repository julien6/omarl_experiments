from typing import List
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from PIL import Image


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class Expenv():

    def __init__(self, env_id: str, num_cpu: int) -> None:
        self.env_id = env_id
        self.num_cpu = num_cpu

        # Create the vectorized environment
        self.vec_env = SubprocVecEnv(
            [make_env(self.env_id, i) for i in range(self.num_cpu)])

        # Stable Baselines provides you with make_vec_env() helper
        # which does exactly the previous steps for you.
        # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
        # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    def train(self):

        eval_callback = EvalCallback(self.vec_env, best_model_save_path="./logs/", verbose=1,
                                     log_path="./logs/", eval_freq=500, deterministic=True, render=False)

        self.model = PPO("MlpPolicy", self.vec_env, verbose=1)
        self.model.learn(total_timesteps=2e10,
                    callback=eval_callback, progress_bar=True)
        del self.model
        # env =gym.make(env_id, render_mode="rgb_array")

    def test(self):

        self.model = PPO.load("./logs/best_model.zip")

        reward_total = [0] * self.num_cpu
        frame_list = []
        
        obs = self.vec_env.reset()
        for _ in range(1000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.vec_env.step(action)
            reward_total = [reward_total[index]+reward for index,
                            reward in enumerate(rewards.tolist())]
            # vec_env.render()
            img = Image.fromarray(self.vec_env.render())
            frame_list.append(img)

        frame_list[0].save(f"{self.env_id}.gif", save_all=True,
                           append_images=frame_list[1:], duration=3, loop=0)

        print("Total rewards: ", reward_total)
        print("Finished")


if __name__ == "__main__":

    # exenv = Expenv("CartPole-v1", 10)
    exenv = Expenv("LunarLander-v2", 10)
    # exenv.train()
    exenv.test()

# ===========


# import gymnasium as gym

# from stable_baselines3 import DQN
# from stable_baselines3.common.evaluation import evaluate_policy
# import numpy as np
# from array2gif import write_gif
# from PIL import Image
# from stable_baselines3.common.callbacks import EvalCallback

# # Create environment
# env = gym.make("LunarLander-v2", render_mode="rgb_array")

# # Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1)

# eval_callback = EvalCallback(env, best_model_save_path="./logs/", verbose=1,
#                              log_path="./logs/", eval_freq=500, deterministic=True, render=False)

# # Train the agent and display a progress bar
# model.learn(total_timesteps=int(2e10),
#             callback=eval_callback, progress_bar=True)

# # Save the agent
# # model.save("dqn_lunar")
# # del model  # delete trained model to demonstrate loading

# # Load the trained agent
# # NOTE: if you have loading issue, you can pass `print_system_info=True`
# # to compare the system on which the model was trained vs the current one
# # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = DQN.load("./logs/best_model.zip", env=env)

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(
#     model, model.get_env(), n_eval_episodes=10)
# print(f"mean : {mean_reward}; std: {std_reward}")

# frame_list = []

# # Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     # vec_env.render("human")

#     img = Image.fromarray(env.render())
#     frame_list.append(img)

# frame_list[0].save("moonlanding.gif", save_all=True,
#                    append_images=frame_list[1:], duration=3, loop=0)

# print("Finished")
