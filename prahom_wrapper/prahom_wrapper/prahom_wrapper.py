import numpy as np
import gymnasium

from custom_envs.movingcompany.moving_company_v0 import env as env_creator, raw_env, parallel_env
from history_model import history_subset
from utils import constraints_integration_mode, gosia_configuration, kosia_configuration
from typing import Any, Callable, Dict, List, Set, Tuple, Union
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType, ParallelEnv
from osh_model import cardinality, deontic_specifications, \
    functional_specifications, group_specifications, link, link_type, obligation, \
    organizational_model, permission, plan, plan_operator, social_preference, social_scheme, \
    structural_specifications, time_constraint_type
from algorithm_configuration import algorithm_configuration, prahom_alg_fac


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
        self.prahom_policy_model = None

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

    def reset(self, seed: int | None = None, options: dict | None = None, prahom_policy_model: bool = False):
        self.prahom_policy_model = None
        print(type(self.env))
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
    # The PRAHOM additional features
    ###########################################

    def train_under_constraints(self, env_creator: Callable[..., Union[AECEnv, ParallelEnv]],
                                osh_model_constraint: organizational_model,
                                constraint_integration_mode: constraints_integration_mode = constraints_integration_mode.CORRECT,
                                algorithm_configuration: algorithm_configuration = prahom_alg_fac.SB3().PPO()) -> None:
        """Restrict history subset to those where any of the given actions is followed by any of the given observations.

        Parameters
        ----------
        env_creator : [..., Union[AECEnv, ParallelEnv]
            The function that enables creating pettingzoo environment either AEC or Parallel

        osh_model_constraint : organizational_model
            The organizational model partially defining organizational specifications to satisfy during training

        constraint_integration_mode : constraint_integration_mode
            The constraint integration mode: CORRECT, PENALIZE, CORRECT_AT_POLICY

        algorithm_configuration : algorithm_configuration
            The configuration of a predefined MARL algorithm (PPO, MADDPG) to be used with a library (StableBaseLines3, RLlib)
            that can be manually fine-tuned or automatically with Hyper-parameter Optimizer (Optuna)

        Returns
        -------
        None

        Examples
        --------
        >>> 

        See Also
        --------
        None
        """
        pass

    def generate_organizational_specifications(self, use_kosia=True, kosia_configuration: kosia_configuration = {},
                                               use_gosia=True, gosia_configuration: gosia_configuration = {}) -> organizational_model:
        """Analyze the trained agents' behavior to determine new organizational specifications.

        Parameters
        ----------
        use_kosia : bool, default True
            Whether using KOSIA to generate organizational specifications

        kosia_configuration: kosia_configuration, default {}
            The KOSIA configuration

        use_gosia : bool, default True
            Whether using GOSIA to generate organizational specifications or complete the KOSIA generated organizational model

        kosia_configuration: gosia_configuration, default {}
            The GOSIA configuration

        Returns
        -------
        organizational_model

        Examples
        --------
        >>> 

        See Also
        --------
        None
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


if __name__ == '__main__':

    pz_env = raw_env()
    pz_env = prahom_wrapper(pz_env)

    pz_env.reset(prahom_policy_model=True)

    pz_env.train_under_constraints(env_creator=env_creator,
                                   osh_model_constraint=organizational_model(
                                       structural_specifications=structural_specifications(
                                           roles={"role_0": history_subset()},
                                           role_inheritance_relations=None,
                                           root_groups=None),
                                       functional_specifications=functional_specifications(
                                           social_scheme=social_scheme(
                                               goals={
                                                   "goal_0": history_subset()},
                                               missions=None,
                                               goals_structure=None,
                                               mission_to_goals=None,
                                               mission_to_agent_cardinality=None),
                                           social_preferences=None),
                                       deontic_specifications=None),
                                   constraint_integration_mode=constraints_integration_mode.CORRECT,
                                   algorithm_configuration=prahom_alg_fac.RLlib().MADDPG()
                                   )

    om = pz_env.generate_organizational_specifications(
        use_kosia=False, use_gosia=True, gosia_configuration=gosia_configuration(generate_figures=True))
    print(om)
