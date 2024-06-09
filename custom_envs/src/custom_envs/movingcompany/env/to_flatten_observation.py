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
from algorithm_configuration import algorithm_configuration
from gym.spaces.utils import flatten_multidiscrete, unflatten_multidiscrete, flatdim_multidiscrete


class to_flatten_observation(BaseWrapper):
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
        return flatdim_multidiscrete(self.env.observation_space(agent))

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)

    def __str__(self) -> str:
        """Returns a name which looks like: "max_observation<space_invaders_v1>"."""
        return f"{type(self).__name__}<{str(self.env)}>"
