from enum import Enum
import numpy as np
import gymnasium

from typing import Any, Callable
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

# Partial Relations with Agent History and Organization Model (PRAHOM)


class ConstraintsRespectMode(Enum):
    """Enum class for policy constraints respect mode.

    The observations received by agents and the actions they make should respect some policy constraints.
    For instance, if an agent is a "follower", it should only receive "order" observation and it should be forced to play some action when it receives one and it should not be able to send one.

    This can be done according to three modes:

        "CORRECT": externally ignore invalid received observations and correct the chosen actions to respect the policy constraints. It enables strictly respecting the policy constraints.

        "CORRECT_AND_PENALIZE": similar to "CORRECT", but with a negative reward for invalid action. It enables teaching to agents to respect policy constraints both for action making and ignoring observations.

        "CORRECT_AT_POLICY": change the agents' policy directly in order to respect policy constraints. It enables changing the action distributions at all steps.
    """
    CORRECT = 0
    CORRECT_AND_PENALIZE = 1
    CORRECT_AT_POLICY = 2


class PrahomWrapper(BaseWrapper):
    """Creates a wrapper around `env` parameter.

    All AECEnv wrappers should inherit from this base class
    """

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType],
                 policy_constraints: Callable = None,
                 policy_constraints_respect_mode: ConstraintsRespectMode = None,
                 generate_organizational_specifications: bool = False,
                 generate_organizational_specifications_figures: bool = False):
        """
        Initializes the wrapper.

        Args:
            env (AECEnv): The environment to wrap.
            policy_constraints (Callable): The policy constraints to respect.
            policy_constraints_respect_mode (ConstraintsRespectMode): The policy constraints respect mode.
            generate_organizational_specifications (bool): Whether to generate organizational specifications.
            generate_organizational_specifications_figures (bool): Whether to generate organizational specifications figures.
        """
        super().__init__()
        self.policy_constraints = policy_constraints
        self.policy_constraints_respect_mode = policy_constraints_respect_mode
        self.organizational_analyze_mode = False
        self.env = env

        # TODO: Ajouter un module SB3 ou RLlib pour en tant qu'option

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

    def generate_organizational_specifications(self):
        pass

    def generate_organizational_specifications_figures(self):
        pass

