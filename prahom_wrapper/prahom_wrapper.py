import copy
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
import gymnasium

from typing import Any, Callable, Dict, List, Set, Tuple
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from custom_envs.movingcompany import moving_company_v0
from history_linked_organizational_model import action, observation, organizational_model, structural_specifications
from train_manager import train_test_manager


# Partial Relations with Agent History and Organization Model (PRAHOM)


class ConstraintsRespectMode(Enum):
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


class history():
    """A class to represent a single agent's history
    """

    def __init__(self, sequence_repr: List[Tuple[observation, action]]) -> None:
        self.sequence_repr = sequence_repr


class history_subset():
    """A class to represent a subset of histories
    """

    def __init__(self, histories: List[history]) -> None:
        self.histories = histories
        self.graph_repr = self.convert_to_graph(histories)

    def convert_to_graph(self, histories: List[history]) -> 'probabilistic_decision_graph':
        weighted_dict: Dict[observation, Dict[action, int]] = {}
        for hist_sequence in histories:
            for obs, act in hist_sequence.sequence_repr:
                if obs not in weighted_dict.keys():
                    weighted_dict[obs] = {}
                if act not in weighted_dict[obs].keys():
                    weighted_dict[obs][act] = 0
                weighted_dict[obs][act] += 1

        probabilistic_dg = copy.deepcopy(weighted_dict)
        for obs, act_weight in weighted_dict.items():
            total_weight = 0
            for act, weight in act_weight.items():
                total_weight += weight
            for act, weight in act_weight.items():
                probabilistic_dg[obs][act] = float(weight/total_weight)

        # graph = {
        #     "nodes": list(weighted_dict.keys()),
        #     "edges": []
        # }
        # for obs, act_weight in weighted_dict.items():
        #     total_weight = 0
        #     for act, weight in act_weight.items():
        #         total_weight += weight
        #     for act, weight in act_weight.items():
        #         graph["edges"] += [(obs, act, float(weight/total_weight))]

        return probabilistic_decision_graph(probabilistic_dg)


class probabilistic_decision_graph():
    """Merge the histories sequences into a single graph where nodes are
        observations and actions are edges weighted by a probability computed
        based on recorded data.
    """

    def __init__(self, graph: Dict[observation, Dict[action, int]]) -> None:
        self.graph = graph

    def next_actions(self, last_observation: observation):
        return self.graph.get(last_observation, None)

    def similarity_percentage(self, last_history: history):
        miss_count = len(last_history.sequence_repr)
        for obs, act in last_history.sequence_repr:
            if obs not in self.graph.keys():
                miss_count -= 1
                continue
            actions = self.graph.get(obs)
            if act not in actions.keys():
                miss_count -= 1
        return float(miss_count/len(last_history.sequence_repr))

    def contains(self, last_history: history):
        for obs, act in last_history.sequence_repr:
            if obs not in self.graph.keys():
                return False
            actions = self.graph.get(obs)
            if act not in actions.keys():
                return False
        return True


class joint_history:
    """A class to represent a joint-history
    """

    def __init__(self, histories: List[history]) -> None:
        self.histories = histories


class joint_history_subset:
    """A class to represent a joint-history with each history as a subset
    """

    def __init__(self, history_subsets: List[history_subset]) -> None:
        self.history_subsets = history_subsets

    def similarity_percentage(self, joint_history: joint_history):
        similarity_percentage = 0
        for index, history_subset in enumerate(self.history_subsets):
            if history_subset is not None:
                similarity_percentage += history_subset.graph_repr.similarity_percentage(
                    joint_history.histories[index])
            else:
                similarity_percentage += 1
        return float(similarity_percentage/len(self.history_subsets))

    def contains(self, joint_history: joint_history):
        for index, history_subset in enumerate(self.history_subsets):
            if history_subset is not None:
                if not history_subset.graph_repr.contains(joint_history.histories[index]):
                    return False
        return True


class joint_histories:
    """A class to represent a set of joint-histories
    """

    def __init__(self, histories: List[joint_history_subset]) -> None:
        self.histories = histories

    def get_history_subset_for_agent(self, agent_index: int) -> probabilistic_decision_graph:
        for joint_history_subset in self.histories:
            for hist_agt_index, history_subset in enumerate(joint_history_subset.history_subsets):
                if hist_agt_index == agent_index and history_subset is not None:
                    return history_subset.graph_repr

    def similarity_percentage(self, joint_history: joint_history):
        similarity_percentage = 0
        for joint_history_subset in self.histories:
            similarity_percentage += joint_history_subset.similarity_percentage(
                joint_history)
        return float(similarity_percentage/len(self.histories))

    def contains(self, joint_history: joint_history):
        for joint_history_subset in self.histories:
            if joint_history_subset.contains(joint_history):
                return True
        return False


class prahom_wrapper(BaseWrapper):
    """Creates a wrapper around `env` parameter.

    All AECEnv wrappers should inherit from this base class
    """

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType],
                 os_to_jt_histories: Dict[str, joint_histories],
                 agent_to_organizational_specification_constraints: Dict[str, str],
                 generate_gosia_figures: bool = False):
        """
        Initializes the wrapper.

        Args:
            env (AECEnv): The environment to wrap.
            os_to_jt_histories (Dict[organizational_model,joint_histories]): The known
                relations from organizational specification to associated histories (expressed in several short ways)
            agent_to_organizational_specification_constraints (Dict[str, organizational_model]): The mapping
                indicating what agent is constrained to what organizational specifications (mostly roles)
            generate_gosia_figures (bool): Whether to generate the GOSIA associated figures helping to better
                understand how the organizational specifications are inferred
        """
        super().__init__(aec_env)
        self.os_to_jt_histories = os_to_jt_histories
        self.agent_to_organizational_specification_constraints = agent_to_organizational_specification_constraints
        self.generate_gosia_figures = generate_gosia_figures
        self.env = env

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
        policy_constraints_respect_mode (ConstraintsRespectMode): The policy constraints respect mode.

        Returns:
        None
        """
        os_cons = [self.agent_to_organizational_specification_constraints[agent]
                   for agent in self.env.possible_agents]
        agent_to_os_joint_histories: Dict[str, probabilistic_decision_graph] = {
            self.agents[agent_index]: self.os_to_jt_histories[organizational_model].get_history_subset_for_agent(agent_index) for agent_index, organizational_model in enumerate(os_cons)}

        self.tt_mngr = train_test_manager(
            train_env, test_env, num_cpu=4, policy_constraints=agent_to_os_joint_histories, mode=mode)
        self.tt_mngr.train(total_step=total_step)

    def generate_organizational_specifications(self):
        """Play all the joint-policies to get joint-histories to apply KOSIA/GOSIA from these
        """

        # Retrieve the total set of joint histories over n_it x n_ep
        joint_hists: List[List[List[Any]]] = self.tt_mngr.test()

        joint_hists = [joint_history([history(agent_hist) for agent_index, agent_hist in enumerate(
            joint_hist)]) for joint_hist in joint_hists]

        print(json.dumps(joint_hists))

        os_kosia = self.kosia(joint_hists)
        os_gosia = self.gosia(os_kosia, joint_hists)

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


if __name__ == "__main__":

    # Dummy example

    # Histories for Role 1
    h11 = history([(0, 1), (0, 2), (2, 3), (3, 4)])
    h12 = history([(10, 1), (1, 7), (6, 3), (7, 4)])
    hs1 = history_subset([h11, h12])  # Agent 1

    # Joint histories for Role 1
    jt_histories_role1 = joint_histories([joint_history_subset([hs1, None, None]),
                                          joint_history_subset(
                                              [None, hs1, None]),
                                          joint_history_subset([None, None, hs1])])

    # Agent 1 's history
    h1 = history([(0, 1), (0, 2), (21, 3), (3, 4)])
    h2 = history([(10, 1), (10, 2), (2, 3), (7, 4)])
    h3 = history([(0, 1), (0, 2), (2, 3), (7, 4)])

    jt_history_agent = joint_history([h1, h2, h3])

    print(jt_histories_role1.contains(jt_history_agent))

    print(jt_histories_role1.get_history_subset_for_agent(0).next_actions(0))

    # role_0 = organizational_model(
    #     structural_specifications=structural_specifications(roles=["role_0"], role_inheritance_relations=None, root_groups=None), functional_specifications=None, deontic_specifications=None)
    # role_1 = organizational_model(
    #     structural_specifications=structural_specifications(roles=["role_1"], role_inheritance_relations=None, root_groups=None), functional_specifications=None, deontic_specifications=None)
    # role_2 = organizational_model(
    #     structural_specifications=structural_specifications(roles=["role_2"], role_inheritance_relations=None, root_groups=None), functional_specifications=None, deontic_specifications=None)

    role_0_hs = history_subset([history([
        ("[0 1 0 0 2 0 0 1 0]", 1),
        ("[0 5 0 0 2 0 0 1 0]", 5),
        ("[0 4 0 0 3 0 0 1 0]", 2),
        ("[0 1 0 0 3 0 0 1 0]", 2),
        ("[0 1 0 0 3 0 0 4 1]", 6),
        ("[0 1 0 0 3 0 0 4 2]", 6),
        ("[0 1 0 0 2 0 0 5 1]", 0),
        ("[0 1 0 0 2 0 0 5 2]", 0),
        ("[0 1 0 0 2 0 0 4 3]", 0),
        ("[0 1 0 0 2 0 0 4 1]", 0)
    ])])

    role_1_hs = history_subset([history([
        ("[1 0 0 5 2 1 0 0 0]", 5),
        ("[2 0 0 5 2 1 0 0 0]", 5),
        ("[1 0 0 4 3 1 0 0 0]", 4),
        ("[2 0 0 4 3 1 0 0 0]", 4),
        ("[0 0 0 1 3 1 0 0 0]", 4),
        ("[0 0 0 1 2 1 0 0 0]", 3),
        ("[0 0 1 1 3 4 0 0 0]", 6),
        ("[0 0 2 1 3 4 0 0 0]", 6),
        ("[0 0 1 1 2 5 0 0 0]", 0),
        ("[0 0 2 1 2 5 0 0 0]", 0),
        ("[1 0 0 4 2 1 0 0 0]", 0),
        ("[3 0 0 4 2 1 0 0 0]", 0),
        ("[0 0 1 1 2 4 0 0 0]", 0),
        ("[0 0 3 1 2 4 0 0 0]", 0)
    ])])

    role_2_hs = history_subset([history([
        ("[0 1 0 0 2 0 1 5 0]", 5),
        ("[0 1 0 0 2 0 2 5 0]", 5),
        ("[0 1 0 0 3 0 1 4 0]", 1),
        ("[0 1 0 0 3 0 2 4 0]", 1),
        ("[0 4 0 0 2 0 0 1 0]", 2),
        ("[0 1 0 0 2 0 0 1 0]", 2),
        ("[0 1 0 0 3 0 0 1 0]", 1),
        ("[0 4 0 0 3 0 0 1 0]", 6),
        ("[0 5 0 0 2 0 0 1 0]", 0),
        ("[0 1 0 0 2 0 1 4 0]", 0),
        ("[0 1 0 0 2 0 3 4 0]", 0)])])

    role_to_jt_histories = {
        "role_0": joint_histories([joint_history_subset([role_0_hs, None, None]),
                                   joint_history_subset(
                                       [None, role_0_hs, None]),
                                   joint_history_subset([None, None, role_0_hs])]),

        "role_1": joint_histories([joint_history_subset([role_1_hs, None, None]),
                                   joint_history_subset(
                                       [None, role_1_hs, None]),
                                   joint_history_subset([None, None, role_1_hs])]),

        "role_2": joint_histories([joint_history_subset([role_2_hs, None, None]),
                                   joint_history_subset(
                                       [None, role_2_hs, None]),
                                   joint_history_subset([None, None, role_2_hs])]),
    }

    agt_to_cons_os = {
        "agent_0": "role_0",
        "agent_1": "role_1",
        "agent_2": "role_2",
    }

    aec_env = moving_company_v0.env()
    aec_env.reset()

    train_env = moving_company_v0.parallel_env(
        render_mode="grid", size=10, seed=42)
    eval_env = moving_company_v0.parallel_env(
        render_mode="rgb_array", size=10, seed=42)

    env = prahom_wrapper(aec_env, role_to_jt_histories, agt_to_cons_os)

    env.train_under_constraints(
        train_env=train_env, test_env=eval_env, total_step=1)

    env.generate_organizational_specifications()
