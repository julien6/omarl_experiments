import math
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
from prahom_wrapper.observable_reward_function import observable_reward_function
from prahom_wrapper.osr_model import deontic_specifications, functional_specifications, time_constraint_type, obligation, organizational_model, social_scheme, structural_specifications
from prahom_wrapper.utils import cardinality, label, history
from pprint import pprint
import numpy as np

normal_leader_adversary_sizes = {
    'self_vel': 2,
    'self_pos': 2,
    'landmark_rel_positions': 10,
    'other_agent_rel_positions': 10,
    'other_agent_velocities': 4,
    'self_in_forest': 2,
    'leader_comm': 4
}

good_sizes = {
    'self_vel': 2,
    'self_pos': 2,
    'landmark_rel_positions': 10,
    'other_agent_rel_positions': 10,
    'other_agent_velocities': 2,
    'self_in_forest': 2
}


def extract_values(concat_array, sizes):
    extracted_values = {}
    index = 0
    for key, size in sizes.items():
        extracted_values[key] = concat_array[index:index+size]
        index += size
    return extracted_values


def dist(pos1, pos2) -> float:
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)


def leader_adversary_opc_fun(history: history, observation: label, agent_name=None) -> List[label]:
    # print("Leader adversary")
    data = extract_values(observation, normal_leader_adversary_sizes)
    other_agent_rel_positions = data["other_agent_rel_positions"]
    # self_pos = (data["self_pos"][0], data["self_pos"][1])
    # other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2] + self_pos[0], other_agent_rel_positions[i*2+1] + self_pos[1]) for i, agent in enumerate(
    #     ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'])}
    other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2], other_agent_rel_positions[i*2+1]) for i, agent in enumerate(
        ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'])}
    # print(other_agent_rel_positions)

    for adversary in ["adversary_0", "adversary_1", "adversary_2"]:
        min_dist = 10000
        min_agent = None
        for good_agent in ["agent_0", "agent_1"]:
            # d = dist(
            #     other_agent_rel_positions[adversary], other_agent_rel_positions[good_agent])
            d = math.sqrt(
                (other_agent_rel_positions[good_agent][0])**2 + (other_agent_rel_positions[good_agent][1])**2)
            if d < min_dist:
                min_dist = d
                min_agent = good_agent
        # target_agent_vec = np.array(other_agent_rel_positions[min_agent]) - np.array(self_pos)
        target_agent_vec = other_agent_rel_positions[min_agent]
        if abs(target_agent_vec[0]) / abs(target_agent_vec[1]) > 1:
            if target_agent_vec[0] > 0:
                return [2]
            return [1]
        else:
            if target_agent_vec[1] > 0:
                return [4]
            return [3]
    return [0]


leader_opc = observable_policy_constraint()
leader_opc.add_custom_function(leader_adversary_opc_fun)


def normal_adversary_opc_fun(history: history, observation: label, agent_name=None) -> List[label]:
    # print("Normal adversary")
    agent_names = ['leadadversary_0', 'adversary_0',
                   'adversary_1', 'adversary_2', 'agent_0', 'agent_1']
    agent_names.remove(agent_name)
    data = extract_values(observation, normal_leader_adversary_sizes)
    other_agent_rel_positions = data["other_agent_rel_positions"]
    other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2], other_agent_rel_positions[i*2+1]) for i, agent in enumerate(
        agent_names)}

    min_dist = 10000
    min_agent = None
    for good_agent in ["agent_0", "agent_1"]:
        d = math.sqrt(
            (other_agent_rel_positions[good_agent][0])**2 + (other_agent_rel_positions[good_agent][1])**2)
        if d < min_dist:
            min_dist = d
            min_agent = good_agent
    target_agent_vec = other_agent_rel_positions[min_agent]
    if abs(target_agent_vec[1]) == 0:
        return [0]
    if float(abs(target_agent_vec[0]) / abs(target_agent_vec[1])) >= 1.:
        if target_agent_vec[0] > 0:
            return [2]
        return [1]
    else:
        if target_agent_vec[1] > 0:
            return [4]
        return [3]
    return [0]


normal_opc = observable_policy_constraint()
normal_opc.add_custom_function(normal_adversary_opc_fun)


def good_opc_fun(history: history, observation: label, agent_name=None) -> List[label]:
    # print("Good agents")
    # pprint(extract_values(observation, good_sizes))
    # print()
    return None


good_opc = observable_policy_constraint()
good_opc.add_custom_function(good_opc_fun)


osr = organizational_model(
    structural_specifications(
        {"r_leader": leader_opc, "r_normal": normal_opc, "r_good": good_opc}, None, None),
    None,
    deontic_specifications(None, {
        obligation("r_leader", None, time_constraint_type.ANY): ["leadadversary_0"],
        obligation("r_normal", None, time_constraint_type.ANY): ["adversary_0", "adversary_1", "adversary_2"],
        obligation("r_good", None, time_constraint_type.ANY): ["agent_0", "agent_1"]}))

env = make_env(environment_name="mpe",
               map_name="simple_world_comm", force_coop=False, organizational_model=osr)


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
