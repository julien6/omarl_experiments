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

import numpy as np
from prahom_wrapper.organizational_model import cardinality, organizational_model, os_encoder
from PIL import Image
from prahom_wrapper.pattern_utils import eval_str_history_pattern, parse_str_history_pattern, history_pattern
from prahom_wrapper.utils import draw_networkx_edge_labels

INFINITY = 'INFINITY'
WILDCARD_NUMBER = 10000


class observation(object):
    """A basic class for any object to be an observation."""
    pass


class action(object):
    """A basic class for any object to be an action."""
    pass


class os_labels(str):
    """A basic class for a subset of organizational specifications."""
    pass


class observation_label(str):
    """A basic string class to represent an observation as a label."""
    pass


class action_label(str):
    """A basic string class to represent an action as a label."""
    pass


class history(List[Tuple[observation_label, action_label]]):
    """A basic class to represent a single agent's history as a list of (observation, action) couples."""
    pass


class history_pattern(str):
    """A basic class to represent a subset of histories matching a sequence pattern."""
    pass


class joint_history(Dict[str, List[Union[observation_label, action_label]]]):
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
                    observation_label, {action_label: {self.ordinal_counter: cardinality(1, 1)}})

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
                    action_label, {observation_label: {self.ordinal_counter: cardinality(1, 1)}})

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
        
        #Convert the pattern into nested list of (sequence,cardinality) couples,
        # where sequence may be a list of labels, or a list of (sequence,cardinality) couples
        # ex: "[obs1,act2,[obs4,act4](0,1)](1,*),obs2,[act2|act4],obs3,act3](1,*)"
        #  -> ([obs1,act2,[obs4,act4](0,1)](1,*),obs2,[act2|act4],obs3,act3], ('1','*'))
        #  -> ([[obs1,act2](1,1),[obs4,act4]](1,*),[obs2,[act2|act4],obs3,act3](1,1)], ('1','*'))
        #  -> ([([obs1,act2],(1,1)),[obs4,act4](0,1)](1,*),[obs2,[act2|act4],obs3,act3](1,1)], ('1','*'))

        # (([*sequence*]),('*cardinaltiy.lowerbound*','*cardinality.upperbound*'))

        def patter_parse(self, string_pattern: str) -> Tuple[List,Tuple[1,1]]:
            

class joint_history_subset:
    """A class to represent each agent's history subset as a joint-history subset."""

    history_graph: Dict[str, Dict[str, int]]


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


class joint_history_factory:

    jh_subset: joint_history_subset

    def __init__(self) -> None:
        self.jh_subset = None

    def new(self) -> 'joint_history_factory':
        self.jh_subset = joint_history_subset()
        return self

    def add_a_history_subset(self, agents_number_among_all: int, history_subsets: List[history_subset]) -> 'joint_history_factory':
        return self

    def create(self) -> joint_history_subset:
        return self.jh_subset


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
    create_relation(self, organizational_model: organizational_model, joint_history_subset: joint_history_subset) -> 'osh_manager':
        Initiates the relation from organizational specifications (under the form of an organizational) to a subset of joint histories.
    """

    osh_graph: Dict[Union[observation_label, action_label],
                    Dict[Union[observation_label, action_label], Dict[os_labels, Dict[int, cardinality]]]]
    """The main graph that represents all (observation,action) and (action,observations) couples of all agents histories decorated with os labels,
    and ordinal number (int) with cardinality."""

    def __init__(self) -> None:
        self.osh_graph = {}

    def create_relation(self, organizational_model: organizational_model, joint_history_subset: joint_history_subset) -> 'osh_manager':
        """Initiates the relation from organizational specifications (under the form of an organizational) to a subset of joint histories.

        Parameters
        ----------
        organizational_model : organizational_model
            An organizational model describing organizational specifications

        joint_history_subset: joint_history_subset
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
                joint_history_subset=JHF.new()
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

    oshr = osh_manager()

    oshr.create_relation(
        organizational_model=OSF.new()
        .add_role("Role_0")
        .create(),
        joint_history_subset=JHF.new()
        .add_a_history_subset(
            agents_number_among_all=1,
            history_subsets=[HF.new()
                             .add_rule("o1", "a1")
                             .add_pattern("[[obs1,act1](1,*)[obs2,act2](0,*)obs3,act4](1,2)")
                             .add_history([("o1", "a1"), ("o2", "a2")])
                             .create()])
        .create()
    )

    # oshr.from_role("Role_0").to_history_subset(hf.new().add_rule("o1", "a1").add_pattern().add_history())
