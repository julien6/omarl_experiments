import random
from typing import Dict
from marllib.envs.base_env.mpe import RLlibMPE
from marllib.marl.common import dict_update, get_model_config, check_algo_type, \
    recursive_dict_update
from marllib.marl.algos import run_il, run_vd, run_cc
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.models import BaseRNN, BaseMLP, CentralizedCriticRNN, CentralizedCriticMLP, ValueDecompRNN, \
    ValueDecompMLP, JointQMLP, JointQRNN, DDPGSeriesRNN, DDPGSeriesMLP
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env
from copy import copy, deepcopy
from tabulate import tabulate
from typing import Any, Dict, Tuple
from os.path import expanduser
import yaml
import os
import sys

from prahom_wrapper.observable_policy_constraint import observable_policy_constraint

class RLlibMPE_action_wrapper(RLlibMPE):

    def __init__(self, env: RLlibMPE, observable_policy_constraint: observable_policy_constraint = None, bonus: int = 10, malus: int = -10):
        self.env = env
        self.opc = observable_policy_constraint
        self.bonus = bonus
        self.malus = malus
        self.last_observation_dict = {}

    def step(self, action_dict):

        modified_actions = deepcopy(action_dict)
        for agent, action in action_dict.items():

            if self.opc is not None:
                obs = self.last_observation_dict[agent]

                if "obs" in list(obs.keys()):
                    obs = str(obs["obs"])

                    home = expanduser("~") + "/observations"
                    with open(home, 'a+') as file:
                        file.write(agent + "\n" + obs + '\n\n' + '='*15 + '\n')

                modified_actions[agent] = {}
                mapped_actions = self.opc.get_actions(None,obs)
                if mapped_actions is not None and len(mapped_actions) > 0:
                    mapped_actions = random.choice(mapped_actions)
                    modified_actions[agent] = mapped_actions
                else:
                    modified_actions[agent] = action

        # Take a step in the environment with modified actions
        obs, rewards, dones, info = self.env.step(modified_actions)

        # Modify rewards based on the action taken
        # for agent, action in action_dict.items():
        #     obs = self.last_observation_dict[agent]

        #     if "obs" in list(obs.keys()):
        #         obs = str(obs["obs"])

        #     if self.opc.get(obs, None) is not None:
        #         mapped_action = self.opc[obs]
        #         if mapped_action == action:
        #             rewards[agent] += self.bonus
        #         else:
        #             pass
        #             #rewards[agent] += self.malus

        self.last_observation_dict = obs

        return obs, rewards, dones, info

    def reset(self):
        self.last_observation_dict = self.env.reset()
        return self.last_observation_dict

    def __getattr__(self, name):
        return getattr(self.env, name)


SYSPARAMs = deepcopy(sys.argv)


def set_ray(config: Dict):
    """
    function of combining ray config with other configs
    :param config: dictionary of config to be combined with
    """
    # default config
    with open(os.path.join(os.path.dirname(local_file), "ray/ray.yaml"), "r") as f:
        ray_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # user config
    user_ray_args = {}
    for param in SYSPARAMs:
        if param.startswith("--ray_args"):
            if "=" in param:
                key, value = param.split(".")[1].split("=")
                user_ray_args[key] = value
            else:  # local_mode
                user_ray_args[param.split(".")[1]] = True

    # update config
    ray_config_dict = dict_update(ray_config_dict, user_ray_args, True)

    for key, value in ray_config_dict.items():
        config[key] = value

    return config


home = expanduser("~")
local_file = f"{home}/miniconda3/envs/marllib/lib/python3.8/site-packages/marllib/marl/__init__.py"


def make_env(
        environment_name: str,
        map_name: str,
        force_coop: bool = False,
        opc: observable_policy_constraint = None,
        **env_params
) -> Tuple[MultiAgentEnv, Dict]:
    """
    construct the environment and register.
    Args:
        :param environment_name: name of the environment
        :param map_name: name of the scenario
        :param force_coop: enforce the reward return of the environment to be global
        :param env_params: parameters that can be pass to the environment for customizing the environment

    Returns:
        Tuple[MultiAgentEnv, Dict]: env instance & env configuration dict
    """

    # default config
    env_config_file_path = os.path.join(os.path.dirname(local_file),
                                        "../envs/base_env/config/{}.yaml".format(environment_name))
    if not os.path.exists(env_config_file_path):
        env_config_file_path = os.path.join(os.path.dirname(local_file),
                                            "../../examples/config/env_config/{}.yaml".format(environment_name))

    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = dict_update(
        env_config_dict["env_args"], env_params, True)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = dict_update(
        env_config_dict["env_args"], user_env_args, True)
    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop

    # combine with exp running config
    env_config = set_ray(env_config_dict)

    # initialize env
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log",
                            "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                env_config["env"]))

    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]

    if env_config["force_coop"]:
        register_env(env_reg_name, lambda _: RLlibMPE_action_wrapper(
            COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"]), opc))
        env = COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    else:
        register_env(env_reg_name, lambda _: RLlibMPE_action_wrapper(
            ENV_REGISTRY[env_config["env"]](env_config["env_args"]), opc))
        env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config
