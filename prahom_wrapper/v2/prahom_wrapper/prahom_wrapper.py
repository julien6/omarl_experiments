import sys

from prahom_wrapper.prahom_wrapper import ConstraintsRespectMode
from prahom_wrapper.v2.prahom_wrapper.history_model import history_subset
sys.path
sys.path.append('../../../.')

from dataclasses import dataclass
from enum import Enum
import numpy as np
import gymnasium

from typing import Any, Callable, Dict, List, Set, Tuple
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from prahom_wrapper.organizational_model import cardinality, deontic_specifications, functional_specifications, group_specifications, link, link_type, obligation, organizational_model, permission, plan, plan_operator, social_preference, social_scheme, structural_specifications, time_constraint_type


class prahom_wrapper(BaseWrapper):
    """Creates a wrapper around `env` parameter.

    All AECEnv wrappers should inherit from this base class
    """

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        """
        Initializes the wrapper.

        Args:
            env (AECEnv): The environment to wrap.
        """
        super().__init__(env)
        self.env = env

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
    
    def set_organizational_analyze_mode(self):
        self.set_organizational_analyze_mode = True
    
    def train_under_constraints(self, env_creator: Callable, osh_model: osh_model,
                                constraint_integration_mode: ConstraintsRespectMode = ConstraintsRespectMode.CORRECT,
                                algorithm_configuration: algorithm_configuration = ALG_MNGR.SB3.default_PPO()) -> None:
        """Restrict history subset to those where any of the given actions is followed by any of the given observations.

        Parameters
        ----------
        env_creator : List[Union[action_label,observation_label]]
            The given source labels

        constraint_integration_mode : List[Union[action_label,observation_label]]
            The given destination labels 

        algorithm_configuration : algorithm_configuration
            The given destination labels 

        Returns
        -------
        None

        Examples
        --------
        >>> hs = history_subset()
        >>> hs.add_labels_to_labels(["act1","act2"], ["obs1","obs2"])

        See Also
        --------
        None
        """
        pass

    def generate_specs(self) -> organizational_model:
        """Play all the joint-policies to get joint-histories to apply KOSIA/GOSIA from these
        """

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
        os_integration_mode = ConstraintsRespectMode.CORRECT,
        algorithm_configuration = algorithm_libraries.SB3().set_algorithm_PPO(configuration)
    )
    env.generate_organizational_specifications(use_gosia = True)
