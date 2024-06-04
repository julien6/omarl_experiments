import sys

from prahom_wrapper.prahom_wrapper import ConstraintsRespectMode
from prahom_wrapper.v2.prahom_wrapper.history_model import history_subset
sys.path
sys.path.append('../../../.')

import copy
from dataclasses import dataclass
from enum import Enum
import json
import time
import numpy as np
import gymnasium

from typing import Any, Callable, Dict, List, Set, Tuple
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from custom_envs.movingcompany import moving_company_v0
from prahom_wrapper.organizational_model import cardinality, deontic_specifications, functional_specifications, group_specifications, link, link_type, obligation, organizational_model, permission, plan, plan_operator, social_preference, social_scheme, structural_specifications, time_constraint_type
from prahom_wrapper.policy_model import joint_policy_constraint
from prahom_wrapper.relation_model import osj_relation
from prahom_wrapper.role_clustering import generate_r_clustering
from prahom_wrapper.train_manager import train_test_manager
from prahom_wrapper.history_model import observation, action
import shutil

from prahom_wrapper.transition_inferring import graph

# Partial Relations with Agent History and Organization Model (PRAHOM)


class prahom_wrapper(BaseWrapper):
    """Creates a wrapper around `env` parameter.

    All AECEnv wrappers should inherit from this base class
    """

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType],
                 os_to_jt_histories: osj_relation,
                 jt_policy_constraint: joint_policy_constraint,
                 label_to_obj: Dict[str, Any],
                 generate_gosia_figures: bool = False):
        """
        Initializes the wrapper.

        Args:
            env (AECEnv): The environment to wrap.
            os_to_jt_histories (osj_relation): The known
                relations from organizational specification to associated histories (expressed in several short ways)
            jt_policy_constraint (joint_policy_constraint): The mapping indicating what agent is constrained to what organizational specifications (mostly roles)
            generate_gosia_figures (bool): Whether to generate the GOSIA associated figures helping to better
                understand how the organizational specifications are inferred
        """
        super().__init__(env)
        self.os_to_jt_histories = os_to_jt_histories
        self.jt_policy_constraint = jt_policy_constraint
        self.generate_gosia_figures = generate_gosia_figures
        self.env = env
        self.label_to_obj: Dict[str, Any] = label_to_obj

        # TODO: Add SB3 or RLlib as an option

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_") and name != "_cumulative_rewards":
            raise AttributeError(
                f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> AECEnv:
        return self.env.unwrapped

    def close(self) -> None:
        self.env.close()

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)

    def state(self) -> np.ndarray:
        return self.env.state()

    def set_organizational_analyze_mode(self):
        self.set_organizational_analyze_mode = True

    def step(self, action: ActionType) -> None:
        self.env.step(action)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)

    def __str__(self) -> str:
        """Returns a name which looks like: "max_observation<space_invaders_v1>"."""
        return f"{type(self).__name__}<{str(self.env)}>"

    ###########################################
    # The PRAHOM additional features:
    ###########################################
    def train_under_constraints(self, train_env, test_env, mode: ConstraintsRespectMode = ConstraintsRespectMode.CORRECT, total_step: int = int(2e10)):
        """
        Create a SB3 PPO MLP model for the training loop integrating the constraints
        following modes. (PRAHOM_osh_training)
        If CORRECT mode is chosen, the actions are corrected on the fly
        If PENALIZE is chosen, the common reward is penalized
        If CORRECT_AT_POLICY is chosen, the policy is modified so actions be corrected 

        Parameters:
        jt_policy_constraints_respect_mode (ConstraintsRespectMode): The policy constraints respect mode.

        Returns:
        None
        """
        # os_cons = [self.jt_policy_constraint[agent]
        #            for agent in self.env.possible_agents]
        # agent_to_os_joint_histories: Dict[str, probabilistic_decision_graph] = {
        #     self.agents[agent_index]: self.os_to_jt_histories[organizational_model].get_history_subset_for_agent(agent_index) for agent_index, organizational_model in enumerate(os_cons)}

        self.tt_mngr = train_test_manager(
            train_env, test_env, label_to_obj=self.label_to_obj, num_cpu=4, policy_constraints=self.jt_policy_constraint, mode=mode)
        self.tt_mngr.train(total_step=total_step)

    def test_trained_model(self):
        """Play all the joint-policies to get joint-histories for assessing
        """
        self.tt_mngr.test()

    def generate_specs(self) -> organizational_model:
        """Play all the joint-policies to get joint-histories to apply KOSIA/GOSIA from these
        """

        # Retrieve the total set of joint histories over n_it x n_ep
        # joint_hists: List[List[List[Any]]] = self.tt_mngr.test()

        # joint_hists = [joint_history([history(agent_hist) for agent_index, agent_hist in enumerate(
        #     joint_hist)]) for joint_hist in joint_hists]

        # print(json.dumps(joint_hists))

        # os_kosia = self.kosia(joint_hists)
        # os_gosia = self.gosia(os_kosia, joint_hists)

        time.sleep(10)

        generate_r_clustering()

        # source = r"./../../assets/images/role_clustering.png"
        # target = r"./role_clustering.png"
        # shutil.copyfile(source, target)

        time.sleep(5)

        # source = r"./../../assets/images/transition_goals.png"
        # target = r"./transition_goals.png"
        # shutil.copyfile(source, target)

        graph()

        time.sleep(2)

        os_gosia = organizational_model(
            structural_specifications=structural_specifications(
                roles=["role_0", "role_1", "role_2"],
                role_inheritance_relations={},
                root_groups={
                    "g1": group_specifications(
                        roles=["role_0", "role_1", "role_2"],
                        sub_groups={}, intra_links=[link("role_0", "role_1", link_type.ACQ), link("role_1", "role_2", link_type.ACQ)],
                        inter_links=[],
                        intra_compatibilities=[],
                        inter_compatibilities=[],
                        role_cardinalities={"role_0": cardinality(1, 1), "role_1": cardinality(1, 1), "role_2": cardinality(1, 1)}, sub_group_cardinalities={})
                }
            ),
            functional_specifications=functional_specifications(
                social_scheme={"sch_1": social_scheme(
                    goals=["goal_1", "goal_2", "goal_3"],
                    missions=["mission_0", "mission_1", "mission_2"],
                    goals_structure=plan(
                        goal='goal_2',
                        sub_goals=['goal_0', 'goal_1'],
                        operator=plan_operator.SEQUENCE,
                        probability=1.0),
                    mission_to_goals={
                        "mission_0": ["goal_0"],
                        "mission_1": ["goal_1"],
                        "mission_2": ["goal_2"]
                    }, mission_to_agent_cardinality={
                        "mission_0": cardinality(1, 1),
                        "mission_1": cardinality(1, 1),
                        "mission_2": cardinality(1, 1),
                    })},
                social_preferences=[]
            ),
            deontic_specifications=deontic_specifications(
                obligations=[obligation("role_0", "mission_0", time_constraint_type.ANY),
                             obligation("role_1", "mission_1",
                                        time_constraint_type.ANY),
                             obligation("role_2", "mission_2", time_constraint_type.ANY)],
                permissions=[permission("role_0", "mission_0", time_constraint_type.ANY),
                             permission("role_1", "mission_1",
                                        time_constraint_type.ANY),
                             permission("role_2", "mission_2", time_constraint_type.ANY)]
            ))

        return os_gosia

    def kosia(self, joint_hists: List[joint_history]) -> organizational_model:
        """The Knowledge-based Organizational Specification Identification Approach
        """
        pass

    def gosia(self, os: organizational_model, joint_hists: List[joint_history]) -> organizational_model:
        """The general Organizational Specification Inference Approach
        """
        pass

    def fusion_organizational_model(self, os_src: organizational_model, os_dest: organizational_model) -> organizational_model:
        """A function to set the organizational specs. from a os model into another (richer) one
        """
        pass

if __name__ == '__main__':

    env = pz_environment_vx()
    env = prahom_wrapper(env)
    env.reset(prahom_policy_model = True)
    env.train_under_constraints(
        os_constraints = (organizational_model(structural_specifications=structural_specifications(roles={"role_0": history_subset()}),
                                                         functional_specifications=functional_specifications(social_scheme=social_scheme(goals={"goal_0": history_subset()})),
                                                         deontic_specifications=None)),
        os_constraints_integration_mode = ConstraintsRespectMode.CORRECT,
        algorithm_configuration = libraries.SB3().PPO_conf()
    )
    env.generate_organizational_specifications(use_gosia = True)
