import sys
sys.path
sys.path.append('../../../.')

import copy
from dataclasses import dataclass, field
import dataclasses
from enum import Enum
import itertools
import json
import random
from typing import Any, Callable, Dict, List, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import re
import numpy as np
from prahom_wrapper.organizational_model import cardinality, organizational_model, os_encoder
from PIL import Image
from prahom_wrapper.utils import draw_networkx_edge_labels


INFINITY = 'INFINITY'
WILDCARD_NUMBER = 10000


class observation(object):
    """A basic class for any object to be an observation."""
    pass


class action(object):
    """A basic class for any object to be an action."""
    pass


class os_label(str):
    """A basic class for a subset of organizational specifications."""
    pass


class observation_label(str):
    """A basic string class to represent an observation as a label."""
    pass


class action_label(str):
    """A basic string class to represent an action as a label."""
    pass


class history_pattern(str):
    """A basic class to represent a subset of histories matching a sequence pattern."""
    pass


class history(List[Tuple[observation_label, action_label]]):
    """A basic class to represent a single agent's history as a list of (observation, action) couples."""
    pass


class joint_history(Dict[str, history]):
    """A basic class to represent a joint-history."""
    pass


class history_subset:
    """A class to represent a set of histories intended to be related to a single subset of organizational specifications for a single agent.
    """

    history_graph: Dict[Union[observation_label, action_label],
                        Dict[Union[observation_label, action_label], Dict[int, cardinality]]]
    ordinal_counter: int

    def __init__(self):
        self.history_graph = {}
        self.ordinal_counter = 0

    def add_observations_to_actions(self, observation_labels: List[observation_label], action_labels: List[action_label]):
        """Restrict history subset to those where any of the given observations is followed by any of the given actions.

        Parameters
        ----------
        observation_labels : List[observation_label]
            The given observation labels

        action_labels : List[action_label]
            The given action labels 

        Returns
        -------
        None

        Examples
        --------
        >>> hs = history_subset()
        >>> hs.add_observations_to_actions(["obs1","obs2"], ["act1","act2"])

        See Also
        --------
        None
        """
        for observation_label in observation_labels:
            for action_label in action_labels:
                self.history_graph.setdefault(
                    observation_label, {})
                self.history_graph[observation_label].setdefault(
                    action_label, {})
                self.history_graph[observation_label][action_label][self.ordinal_counter] = cardinality(
                    1, 1)
        self.ordinal_counter += 1

    def add_actions_to_observations(self, action_labels: List[action_label], observation_labels: List[observation_label]):
        """Restrict history subset to those where any of the given actions is followed by any of the given observations.

        Parameters
        ----------
        action_labels : List[action_label]
            The given action labels

        observation_labels : List[observation_label]
            The given observation labels 

        Returns
        -------
        None

        Examples
        --------
        >>> hs = history_subset()
        >>> hs.add_actions_to_observations(["act1","act2"], ["obs1","obs2"])

        See Also
        --------
        None
        """
        for observation_label in observation_labels:
            for action_label in action_labels:
                self.history_graph.setdefault(
                    action_label, {})
                self.history_graph[action_label].setdefault(
                    observation_label, {})
                self.history_graph[action_label][observation_label][self.ordinal_counter] = cardinality(
                    1, 1)
        self.ordinal_counter += 1

    def add_labels_to_labels(self, src_labels: List[Union[action_label, observation_label]],
                             dst_labels: List[Union[action_label, observation_label]],
                             src_to_dst_cardinality: cardinality = cardinality(1, 1)):
        """Restrict history subset to those where any of the given actions is followed by any of the given observations.

        Parameters
        ----------
        src_labels : List[Union[action_label,observation_label]]
            The given source labels

        dst_labels : List[Union[action_label,observation_label]]
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
        for src_label in src_labels:
            for dst_label in dst_labels:
                self.history_graph.setdefault(
                    src_label, {})
                self.history_graph[src_label].setdefault(
                    dst_label, {})
                self.history_graph[src_label][dst_label][self.ordinal_counter] = src_to_dst_cardinality
        self.ordinal_counter += 1

    def add_history(self, history: history):
        """Add the given history in the history subset.

        Parameters
        ----------
        history : history
            The given history

        Returns
        -------
        None

        Examples
        --------
        >>> hs = history_subset()
        >>> hs.add_history([("obs1","act1"),("obs2","act2")])

        See Also
        --------
        None
        """
        last_act = None
        for obs, act in history:
            if last_act is not None:
                self.add_actions_to_observations([last_act], [obs])
            self.add_observations_to_actions([obs], [act])
            last_act = act

    def add_pattern(self, history_pattern: history_pattern):
        """Restrict history subset to those matching the given sequence pattern.

        Parameters
        ----------
        history : str
            The given sequence pattern

        Returns
        -------
        None

        Examples
        --------
        >>> hs = history_subset()
        >>> hs.add_pattern("[obs1,act2,[obs4,act4](0,1)](1,*),obs2,[act2|act4],obs3,act3](1,*)")

        See Also
        --------
        None
        """

        def uniformize(string_pattern: str) -> str:

            regex = r'\[([\",0-9A-Za-z]{2,}?),[\(\[]|\),([\",0-9A-Za-z]{2,}?)\],\('
            matches = re.findall(regex, string_pattern)

            for group in matches:
                group = group[0] if group[0] != "" else group[1]
                if group != "":
                    string_pattern = string_pattern.replace(
                        group, f"([{group}],('1','1'))")
            return string_pattern

        def convert_to_tuple(string_pattern: str) -> Tuple[List, cardinality]:
            stack = []
            i = 0
            while i < len(string_pattern):
                character = string_pattern[i]
                if character == "[":
                    stack += ["["]
                elif character == "]":
                    stack[-1] += "]"
                    if string_pattern[i+1] == "(":
                        card = ""
                        i += 1
                        while i < len(string_pattern):
                            char_card = string_pattern[i]
                            card += char_card
                            if char_card == ")":
                                break
                            i += 1
                        sequence = stack.pop()
                        sequence = f'({sequence},{card})'
                        if (i == len(string_pattern) - 1):
                            seq = sequence.replace("[", "[\"").replace("]", "\"]") \
                                .replace("(", "(\"").replace(")", "\")").replace(",", "\",\"") \
                                .replace("\"(", "(").replace(")\"", ")") \
                                .replace("\"[", "[").replace("]\"", "]")
                            seq = uniformize(seq)
                            return eval(seq)
                        stack[-1] += sequence
                else:
                    stack[-1] += character
                i += 1

        tuple_pattern = convert_to_tuple(history_pattern)

        def is_only_labels(tuple_pattern: Tuple[List, cardinality]) -> bool:
            label_or_tuple_list, card = tuple_pattern
            for label_or_tuple in label_or_tuple_list:
                if type(label_or_tuple) == tuple:
                    return False
            return True

        self.last_label = None

        def parse_into_graph(tuple_pattern: Tuple[List, cardinality]) -> None:

            if is_only_labels(tuple_pattern):

                labels_sequence, labels_card = tuple_pattern

                for label in labels_sequence:
                    if self.last_label is None:
                        self.last_label = label
                    else:
                        self.add_labels_to_labels([self.last_label], [label])
                        self.last_label = label
                if not (labels_card[0] == "1" and labels_card[1] == "1"):
                    self.add_labels_to_labels([self.last_label], [labels_sequence[0]],
                                              src_to_dst_cardinality=cardinality(labels_card[0], labels_card[1]))
                return labels_sequence[0]

            else:
                tuples_sequence, sequences_card = tuple_pattern
                start_label = copy.copy(self.last_label)
                for i, sub_tuple in enumerate(tuples_sequence):
                    sl = parse_into_graph(sub_tuple)
                    if i == 0:
                        start_label = sl
                if not (sequences_card[0] == "1" and sequences_card[1] == "1"):
                    self.add_labels_to_labels([self.last_label], [start_label],
                                              src_to_dst_cardinality=cardinality(sequences_card[0], sequences_card[1]))

        parse_into_graph(tuple_pattern)

    def plot_graph(self, show: bool = False, render_rgba: bool = False, save: bool = False):

        # Create a directed graph object
        G = nx.DiGraph()

        oriented_edges = {}

        for src_label in self.history_graph:
            for dst_label in self.history_graph[src_label]:
                G.add_edge(src_label, dst_label)
                if (src_label, dst_label) in oriented_edges.keys():
                    return
                oriented_edges[(src_label, dst_label)] = str(
                    self.history_graph[src_label][dst_label])

        # Draw the graph
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)

        curved_edges = [
            edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
        arc_rad = 0.25
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

        curved_edge_labels = {edge: oriented_edges[(
            f'{edge[0]}', f'{edge[1]}')] for edge in curved_edges}
        straight_edge_labels = {edge: oriented_edges[(
            f'{edge[0]}', f'{edge[1]}')] for edge in straight_edges}
        draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

        # Show the plot
        if (save):
            fig.savefig(f"{random.randint(1, 100)}.png",
                        bbox_inches='tight', pad_inches=0)
        if (show):
            plt.show()

        if (render_rgba):
            # fig.canvas.draw()
            # return Image.frombytes('RGB',
            # fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

            fig.canvas.draw()  # Draw the canvas, cache the renderer
            image_flat = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            # NOTE: reversed converts (W, H) from get_width_height to (H, W)
            image = image_flat.reshape(
                *reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)
            return image


class joint_history_subset(Dict[str, history_subset]):
    """A basic class to represent a joint history subsets."""
    pass


class os_to_history_subset_relations:
    """A class to represent links between some organizational specifications and history subsets."""

    history_graph: Dict[Union[observation_label, action_label],
                        Dict[Union[observation_label, action_label], Dict[os_label, Dict[int, cardinality]]]]

    def __init__(self):
        self.history_graph = {}

    def add_os_to_history_subset_relation(self, os_label: os_label, history_subsets: List[history_subset]):
        """Add a relation between organizational specifications to corresponding history subsets.

        Parameters
        ----------
        os_label : os_label
            The given organizational specification label

        history_subsets: List[history_subset]
            The history subsets related to given organizational specifications

        Returns
        -------
        None

        Examples
        --------
        Create a couple between a role that can be related to at least one agent among all.
        That couple is associated to a simple history subset.
        >>> h = history_subset()
        >>> h.add_actions_to_observations(["act1"],["obs1"])
        >>> hs = os_to_history_subset_relations()
        >>> hs.add_os_to_history_subset_relation("role_0", [h])

        See Also
        --------
        None
        """
        for agent_index, history_subset in enumerate(history_subsets):
            for src_label in history_subset.history_graph:
                self.history_graph.setdefault(src_label, {})
                for dst_label in history_subset.history_graph[src_label]:
                    self.history_graph[src_label].setdefault(dst_label, {})
                    os_agent_label = os_label + "_" + \
                        "{" + str(agent_index) + "/" + \
                        str(len(history_subsets)) + "}"
                    self.history_graph[src_label][dst_label].setdefault(
                        os_agent_label, {})
                    ordinal_number_to_cardinality = copy.deepcopy(
                        history_subset.history_graph[src_label][dst_label])
                    self.history_graph[src_label][dst_label][os_agent_label].update(ordinal_number_to_cardinality)

    def get_os_from_joint_history(self, joint_history: joint_history) -> Dict[str, List[os_label]]:
        """Get the organizational specifications from a joint-history.

        Parameters
        ----------
        joint_history : joint_history
            The given joint-history

        Returns
        -------
        Dict[str, List[os_label]]
            `Dict[str, List[os_label]]` the organizational specifications associated with agent

        Examples
        --------
        Get organizational specifications from a simple joint-history
        >>> h = history_subset()
        >>> jh = {"agent_1": copy.deepcopy(h), "agent_2": copy.deepcopy(h)}
        >>> h.add_actions_to_observations(["act1"],["obs1"])
        >>> jh["agent_0"] = h
        >>> hs = os_to_history_subset_relations()
        >>> hs.add_os_to_history_subset_relation("role_0", [h])
        >>> hs.get_os_from_joint_history(jh)
        ["agent_0": "role_0", "agent_1": None, "agent_2": None]

        See Also
        -------
        None
        """

        joint_matching_os_labels: Dict[str, List[os_label]] = {}
        for agent, history in joint_history.items():
            auxiliary_history_graph: Dict[Union[observation_label, action_label],
                        Dict[Union[observation_label, action_label], Dict[os_label, Dict[int, cardinality]]]] = {}
            ordinal_number = 0
            os_labels = []
            for label_transition in history:
                src_label, dst_label = label_transition
                
                os_transition_labels = list(self.history_graph.get(src_label, {}).get(dst_label, {}).keys())

                for osl in os_label:

                    ordinal_numbers = list(self.history_graph[src_label][dst_label][osl].keys())

                    if ordinal_number in ordinal_numbers:

                        card = self.history_graph[src_label][dst_label][osl][ordinal_number]
                        if not(card.lower_bound == 1 and card.upper_bound == 1):
                            auxiliary_history_graph.setdefault(src_label, {})
                            auxiliary_history_graph[src_label].setdefault(dst_label, {})
                            auxiliary_history_graph[src_label].setdefault(dst_label, {})
                            ordinal_number = None
                        
                    ordinal_number += 1

            joint_matching_os_labels[agent] = list(matching_os_labels)

        return joint_matching_os_labels

    def get_joint_history_subsets_from_os(self, os_label: os_label) -> List[joint_history_subset]:
        """Get the joint-history subsets associated with organizational specifications.

        Parameters
        ----------
        os_label : os_label
            The given organizational specification label

        Returns
        -------
        List[joint_history_subset]
            `List[joint_history_subset]` the joint-history subsets associated with given organizational specification label

        Examples
        --------
        Get organizational specifications from a simple joint-history
        >>> h = history_subset()
        >>> jh = {"agent_1": copy.deepcopy(h), "agent_2": copy.deepcopy(h)}
        >>> h.add_actions_to_observations(["act1"],["obs1"])
        >>> jh["agent_0"] = h
        >>> hs = os_to_history_subset_relations()
        >>> hs.add_os_to_history_subset_relation("role_0", [h])
        >>> hs.get_joint_history_subset_from_os("role_0")

        See Also
        -------
        None
        """

        agents_hs: Dict[str, Dict[Union[observation_label, action_label],
                        Dict[Union[observation_label, action_label], Dict[int, cardinality]]]] = {}
        for src_label in self.history_graph:
            for dst_label in self.history_graph[src_label]:
                os_labels = self.history_graph[src_label][dst_label].keys()
                for osl in os_labels:
                    if os_label in osl:
                        agents_hs.setdefault(osl, {})
                        agents_hs[osl].setdefault(src_label, {})
                        agents_hs[osl][src_label].setdefault(dst_label, {})
                        agents_hs[osl][src_label][dst_label] = copy.deepcopy(
                            self.history_graph[src_label][dst_label][os_label])

        return list(agents_hs.values())

    def get_next_actions_from_observation_and_os(self, joint_history: joint_history, \
                                                 observation_label: observation_label, os_label: os_label) -> List[action_label]:
        """According to organizational specifications, get the next actions that should be played when a joint-history has already been played and an observation is received.

        Parameters
        ----------
        joint_history: joint_history
            The joint-history that has already been played

        observation_label : observation_label
            The received observation label
        
        os_label : os_label
            The given organizational specification label

        Returns
        -------
        List[action_label]
            `List[action_label]` the action labels that can be chosen according to given os and played joint-history

        Examples
        --------
        Get organizational specifications from a simple joint-history
        >>> h = history_subset()
        >>> jh = {"agent_1": copy.deepcopy(h), "agent_2": copy.deepcopy(h)}
        >>> h.add_actions_to_observations(["act1"],["obs1"])
        >>> jh["agent_0"] = h
        >>> hs = os_to_history_subset_relations()
        >>> hs.add_os_to_history_subset_relation("role_0", [h])
        >>> hs.get_joint_history_subset_from_os("role_0")

        See Also
        -------
        None
        """
        for action_label in self.history_graph.get(observation_label, {}):
            if os_label in self.history_graph[observation_label][action_label]:
                return action_label
        return None


class os_factory:

    os_model: organizational_model

    def __init__(self) -> None:
        self.os_model = None

    def new(self):
        self.os_model = organizational_model(
            structural_specifications=None, functional_specifications=None, deontic_specifications=None)
        return self

    def add_role(self, role_name: str) -> 'os_factory':
        self.os_model.structural_specifications.roles.append(role_name)
        return self

    def create(self) -> organizational_model:
        return self.os_model


class history_factory:
    h_subset: history_subset

    def __init__(self):
        self.h_subset = history_subset()

    def new(self) -> 'history_factory':
        return self

    def add_rule(self, observation: observation_label, action: action_label) -> 'history_factory':
        return self

    def add_pattern(self, str_patter: str) -> 'history_factory':
        return self

    def add_history(self, history: history) -> 'history_factory':
        return self

    def create(self) -> history_subset:
        return self.h_subset


class joint_history_factory:

    jh_subset: os_to_history_subset_relations

    def __init__(self) -> None:
        self.jh_subset = None

    def new(self) -> 'joint_history_factory':
        self.jh_subset = os_to_history_subset_relations()
        return self

    def add_a_history_subset(self, agents_number_among_all: int, history_subsets: List[history_subset]) -> 'joint_history_factory':
        return self

    def create(self) -> os_to_history_subset_relations:
        return self.jh_subset


OSF = os_factory()
"""The main instance of Organizational Specification Factory"""

HF = history_factory()
"""The main instance of History Factory"""

JHF = joint_history_factory()
"""The main instance of Joint-History Factory"""


class osh_manager():
    """
    A class to represent the links between organizational specifications and all possible joint-histories when applied for any agent subset.

    Attributes
    ----------
    osh_grah : Dict[Any,Any]
        The graph to represent all joint-histories whose edges are decorated with their respective os labels

    Methods
    -------
    create_relation(self, organizational_model: organizational_model, os_to_history_subset_relations: os_to_history_subset_relations) -> 'osh_manager':
        Initiates the relation from organizational specifications (under the form of an organizational) to a subset of joint histories.
    """

    osh_graph: Dict[Union[observation_label, action_label],
                    Dict[Union[observation_label, action_label], Dict[os_label, Dict[int, cardinality]]]]
    """The main graph that represents all (observation,action) and (action,observations) couples of all agents histories decorated with os labels,
    and ordinal number (int) with cardinality."""

    def __init__(self) -> None:
        self.osh_graph = {}

    def create_relation(self, organizational_model: organizational_model, os_to_history_subset_relations: os_to_history_subset_relations) -> 'osh_manager':
        """Initiates the relation from organizational specifications (under the form of an organizational) to a subset of joint histories.

        Parameters
        ----------
        organizational_model : organizational_model
            An organizational model describing organizational specifications

        os_to_history_subset_relations: os_to_history_subset_relations
            A subset of joint-histories describing how a subset of agent adopting the organizational specifications should behave

        Returns
        -------
        osh_manager
            `osh_manager` of the current instance

        Examples
        --------
        >>> oshr = osh_manager()

        >>> oshr.create_relation(
                organizational_model=OSF.new()
                .add_role("Role_0")
                .create(),
                os_to_history_subset_relations=JHF.new()
                .add_a_history_subset(
                    agents_number_among_all=1,
                    history_subsets=[HF.new()
                                    .add_rule("o1", "a1")
                                    .add_pattern("[[obs1,act1](1,*)[obs2,act2](0,*)obs3,act4](1,2)")
                                    .add_history([("o1", "a1"), ("o2", "a2")])
                                    .create()])
                .create()
            )

        See Also
        --------
        """
        return self


if __name__ == '__main__':

    hs = history_subset()
    # hs.add_pattern("[obs1,[act2,obs2](1,2),act3](1,*)")
    hs.add_pattern(
        "[obs1,act1,[obs2,act2,[obs3,act3](2,2),[obs14,[act45,obs78](0,*),act15](14,12),[obs3,act3](2,2),obs4,act4](1,2)](1,*)")
    # hs.plot_graph(show=True)

    hs = history_subset()
    hs.add_actions_to_observations(["obs1"], ["act1"])
    hs.add_actions_to_observations(["obs2"], ["act2"])
    # hs.plot_graph(show=True)

    hs = history_subset()
    hs.add_history([("obs1", "act1"), ("obs2", "act2"), ("obs3", "act3")])
    # hs.plot_graph(show=True)

    # oshr = osh_manager()

    # oshr.create_relation(
    #     organizational_model=OSF.new()
    #     .add_role("Role_0")
    #     .create(),
    #     os_to_history_subset_relations=JHF.new()
    #     .add_a_history_subset(
    #         agents_number_among_all=1,
    #         history_subsets=[HF.new()
    #                          .add_rule("o1", "a1")
    #                          .add_pattern("[[obs1,act1](1,*)[obs2,act2](0,*)obs3,act4](1,2)")
    #                          .add_history([("o1", "a1"), ("o2", "a2")])
    #                          .create()])
    #     .create()
    # )

    # oshr.from_role("Role_0").to_history_subset(hf.new().add_rule("o1", "a1").add_pattern().add_history())
