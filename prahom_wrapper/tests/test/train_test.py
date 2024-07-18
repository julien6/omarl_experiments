import sys
import os
from typing import List

from marllib import marl
from ray.tune import Analysis
from pathlib import Path
from datetime import datetime

from action_constrain_wrapper import RLlibMPE_action_wrapper, make_env
from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl import register_env

from marllib.envs.base_env.mpe import REGISTRY
from ray.tune.registry import _global_registry
from prahom_wrapper.observable_policy_constraint import observable_policy_constraint
from prahom_wrapper.utils import label, history


opc = observable_policy_constraint()


def manual_next_action(history: history, observation_label: label) -> List[label]:
    import random
    # return [1, 2] if random.random() < 0.25 else [3, 1] if random.random() < 0.25 else [0]
    return [1,2] if random.random() < 0.2 else None

opc.add_custom_function(manual_next_action)


env = make_env(environment_name="mpe",
               map_name="simple_world_comm", force_coop=False, opc=opc)


# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")


# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

checkpoint_freq = 500

if len(sys.argv) > 1 and sys.argv[1] == "--test":

    checkpoint_path = None

    mode = "max"
    metric = 'episode_reward_mean'

    algorithm = mappo.name
    map_name = env[1]["env_args"]["map_name"]
    arch = model[1]["model_arch_args"]["core_arch"]
    running_directory = '_'.join([algorithm, arch, map_name])
    running_directory = f"./exp_results/{running_directory}"

    if (os.path.exists(running_directory)):

        # Trouver tous les fichiers de checkpoint dans le dossier
        checkpoint_trial_folders = [f for f in os.listdir(
            running_directory) if f.startswith(algorithm.upper())]

        # 2024-06-16_15-38-06
        date_format = '%Y-%m-%d_%H-%M-%S'

        checkpoint_trial_folders.sort(
            key=lambda f: datetime.strptime(str(f[-19:]), date_format))

        checkpoint_path = os.path.join(
            running_directory, checkpoint_trial_folders[-1])

    analysis = Analysis(
        checkpoint_path, default_metric=metric, default_mode=mode)
    df = analysis.dataframe()

    idx = df[metric].idxmax()

    training_iteration = df.iloc[idx].training_iteration

    best_logdir = df.iloc[idx].logdir

    best_checkpoint_dir = [p for p in Path(best_logdir).iterdir(
    ) if "checkpoint_" in p.name and (int(p.name.split("checkpoint_")[1]) <= training_iteration and training_iteration <= int(p.name.split("checkpoint_")[1]) + checkpoint_freq)][0]

    checkpoint_number = str(
        int(best_checkpoint_dir.name.split("checkpoint_")[1]))
    best_checkpoint_file_path = os.path.join(
        best_checkpoint_dir, f'checkpoint-{checkpoint_number}')

    # rendering
    mappo.render(env, model,
                 restore_path={'params_path': f"{checkpoint_path}/params.json",  # experiment configuration
                               'model_path': best_checkpoint_file_path,  # checkpoint path
                               'render': True},  # render
                 local_mode=True,
                 share_policy="group",
                 checkpoint_end=False)

else:

    # start learning + extra experiment settings if needed. remember to check ray.yaml before use
    mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=1,
              num_workers=10, share_policy='group', checkpoint_freq=checkpoint_freq)

    # mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0,
    #           num_workers=1, share_policy='group', checkpoint_freq=checkpoint_freq)
