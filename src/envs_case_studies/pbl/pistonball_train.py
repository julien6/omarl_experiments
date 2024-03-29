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


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space,
                              num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = pistonball_v6.parallel_env(
        n_pistons=10,
        time_penalty=-0.1,
        continuous=False,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    env.reset(42)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    env_name = "pistonball_v6"

    register_env(env_name, lambda config: ParallelPettingZooEnv(
        env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    local_dir = os.path.dirname(os.path.abspath(__file__))

    configuration = config.to_dict()

    # Restauration d'un checkpoint
    # Le chemin vers le dossier où les checkpoints sont stockés
    checkpoint_dir = os.path.join(local_dir, "ray_results", env_name, "PPO")

    analysis = None

    if (os.path.exists(checkpoint_dir)):

        # Trouver tous les fichiers de checkpoint dans le dossier
        checkpoint_trial_folders = [f for f in os.listdir(
            checkpoint_dir) if f.startswith("PPO_")]

        date_format = '%Y-%m-%d_%H-%M-%S'

        # Trier les fichiers de checkpoint par numéro de checkpoint
        # PPO_pistonball_v6_3594a_00000_0_2024-02-29_14-19-52
        checkpoint_trial_folders.sort(
            key=lambda f: datetime.strptime(str(f[-19:]), date_format))

        checkpoint_dir = os.path.join(
            checkpoint_dir, checkpoint_trial_folders[-1])

        # Trouver tous les fichiers de checkpoint dans le dossier
        checkpoint_files = [f for f in os.listdir(
            checkpoint_dir) if f.startswith("checkpoint_")]

        # Trier les fichiers de checkpoint par numéro de checkpoint
        checkpoint_files.sort(key=lambda f: int(f.split("_")[1]))

        # Le chemin vers le dernier fichier de checkpoint
        last_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])

        print("Last checkpoint path: ", last_checkpoint)

        analysis = tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": 5000000 if not os.environ.get(
                "CI") else 50000},
            checkpoint_freq=1,
            local_dir=local_dir + "/ray_results/" + env_name,
            config=configuration,
            restore=last_checkpoint
        )
    else:

        analysis = tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": 5000000 if not os.environ.get(
                "CI") else 50000},
            checkpoint_freq=1,
            local_dir=local_dir + "/ray_results/" + env_name,
            config=configuration
        )

    # print(analysis)

    # # Obtenez le chemin vers le checkpoint du meilleur essai
    # best_trial = analysis.get_best_trial("episode_reward_mean")
    # best_checkpoint = best_trial.checkpoint.value
    # print(best_checkpoint)

    # # Specify the path to the directory containing checkpoint files
    # checkpoint_dir = "./ray_results/pistonball_v6/PPO/PPO_pistonball_v6_8ef6f_00000_0_2024-03-01_16-58-34/"

    # # Load the Analysis object
    # analysis = tune.ExperimentAnalysis(checkpoint_dir)

    # # Access various information from the analysis object
    # episode_rewards = analysis.dataframe["episode_reward_mean"]
    # iterations = analysis.dataframe["training_iteration"]

    # # Plot the rewards over training iterations
    # import matplotlib.pyplot as plt

    # plt.plot(iterations, episode_rewards)
    # plt.xlabel("Training Iteration")
    # plt.ylabel("Mean Episode Reward")
    # plt.title("Training Progress")
    # plt.show()