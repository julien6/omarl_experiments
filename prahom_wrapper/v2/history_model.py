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
    pass


class action(object):
    pass


class observation_label(str):
    pass


class action_label(str):
    pass


class action_label(str):
    pass


class history(List[Tuple[observation_label, action_label]]):
    pass


class history_pattern(str):
    pass


class joint_history(Dict[str, List[Union[observation_label, action_label]]]):
    pass


class history_subset:

    history_graph: Dict[str, Dict[str, int]]
    history_number: int

    def __init__(self):
        self.history_graph = {}
        self.history_number = -1

    def add_observation_action(self, obs: observation_label, act: action_label):
        self.history_graph.setdefault(obs, {act: 0})
        self.history_graph[obs][act] += 1

    def add_action_observation(self, act: action_label, obs: observation_label):
        self.history_graph.setdefault(act, {obs: 0})
        self.history_graph[act][obs] += 1

    def add_history(self, history: history):
        last_obs = None
        last_act = None
        for obs, act in history:
            self.add
            last_obs = obs
            last_act = act

    def add_pattern(self, history_pattern: history_pattern):
        pass


class joint_history_subset:
    history_graph: Dict[str, Dict[str, int]]


class osh_relations():

    def from_os(self, os_model: organizational_model) -> 'osh_relations':
        return self

    def to_joint_history_subset(self,) -> 'osh_relations':
        return self


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

    def add_a_history_subset(self, agents_number_among_all: int, history_subset: history_subset) -> 'joint_history_factory':
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
HF = history_factory()
JHF = joint_history_factory()

if __name__ == '__main__':

    oshr = osh_relations()

    # 
    oshr.from_os(OSF.new().add_role("Role_0").create())\
        .to_joint_history_subset(
            JHF.new().add_a_history_subset(
                agents_number_among_all=1,
                history_subset=HF.new().add_rule("o1", "a1").add_pattern(
                    "ok").add_history([("o1", "a1"), ("o2", "a2")]).create()
            ).create()
    )

    # oshr.from_role("Role_0").to_history_subset(hf.new().add_rule("o1", "a1").add_pattern().add_history())
