"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from pettingzoo.butterfly import pistonball_v6
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    env_name = "pistonball_v6"

    local_dir = os.path.dirname(os.path.abspath(__file__))

    # Restauration d'un checkpoint
    # Le chemin vers le dossier où les checkpoints sont stockés
    checkpoint_dir = os.path.join(local_dir, "ray_results", env_name, "PPO")

    # # Trouver tous les fichiers de checkpoint dans le dossier
    # checkpoint_trial_folders = [f for f in os.listdir(
    #     checkpoint_dir) if f.startswith("PPO_")]

    # date_format = '%Y-%m-%d_%H-%M-%S'

    # # Trier les fichiers de checkpoint par numéro de checkpoint
    # # PPO_pistonball_v6_3594a_00000_0_2024-02-29_14-19-52
    # checkpoint_trial_folders.sort(
    #     key=lambda f: datetime.strptime(str(f[-19:]), date_format))

    # checkpoint_dir = os.path.join(
    #     checkpoint_dir, checkpoint_trial_folders[-1])

    # Load the Analysis object
    analysis = tune.ExperimentAnalysis(checkpoint_dir)

    print(analysis)

    # Obtenez le chemin vers le checkpoint du meilleur essai
    best_trial = analysis.get_best_trial("episode_reward_mean")
    best_checkpoint = best_trial.checkpoint
    print("!!! ", best_checkpoint)

    # Access various information from the analysis object
    # episode_rewards = analysis.dataframe("episode_reward_mean")
    # iterations = analysis.dataframe("training_iteration")
    episode_rewards = analysis.trial_dataframes
    episode_rewards = episode_rewards[list(episode_rewards.keys())[0]]
    # rewards = episode_rewards["hist_stats/episode_reward"].values.tolist()[0]
    # rewards = episode_rewards["episode_reward_mean"].values.tolist()[0]
    rewards = episode_rewards["episode_reward_mean"].values.tolist()
    iterations = range(0, len(rewards))

    print(rewards)
    print(iterations)

    # Plot the rewards over training iterations
    plt.plot(iterations, rewards)
    plt.xlabel("Training Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.title("Training Progress")
    plt.savefig("training_progress.png")
    plt.close()
