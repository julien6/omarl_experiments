from enum import Enum


class constraints_integration_mode(Enum):
    """Enum class for policy constraints respect mode.

    The observations received by agents and the actions they make should respect some policy constraints.
    For instance, if an agent is a "follower", it should only receive "order" observation and it should be forced to play some action when it receives one and it should not be able to send one.

    This can be done according to three modes:

        "CORRECT": externally ignore invalid received observations and correct the chosen actions to respect the policy constraints. It enables strictly respecting the policy constraints.

        "PENALIZE": a negative reward for invalid action. It enables teaching to agents to respect policy constraints both for action making and ignoring observations.

        "CORRECT_AT_POLICY": change the agents' policy directly in order to respect policy constraints. It enables changing the action distributions at all steps.
    """
    CORRECT = 0
    PENALIZE = 1
    CORRECT_AT_POLICY = 2
