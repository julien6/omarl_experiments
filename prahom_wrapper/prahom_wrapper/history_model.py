import numpy as np
import copy
import dataclasses
import itertools
import json
import random
import networkx as nx
import matplotlib.pyplot as plt
import re

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, OrderedDict, Tuple, Union
from pprint import pprint
from utils import cardinality, draw_networkx_edge_labels
from PIL import Image
from history_graph import history_graph, observation, action, label, history, history_pattern


class history_subset:

    def __init__(self) -> None:
        self.custom_functions: List[Callable[[
            history, observation], List[action]]] = []
        self.history_graphs: List[history_graph] = []

    def add_pattern(self, pattern: history_pattern, label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act: Dict[label,
                                    Union[observation, action]] = label_to_obs_act
        self.history_graphs += [
            self._convert_pattern_to_history_graph(pattern)]

    def add_rules(self, rules: Dict[Tuple[history, observation], List[action]], label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act: Dict[label,
                                    Union[observation, action]] = label_to_obs_act
        self.history_graphs += [self._convert_rules_to_history_graph(rules)]

    def add_history(self, history: history, label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act: Dict[label,
                                    Union[observation, action]] = label_to_obs_act
        self.history_graphs += [
            self._convert_history_to_history_graph(history)]

    def add_custom_function(self, custom_function: Callable[[history, observation], List[action]], label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act: Dict[label,
                                    Union[observation, action]] = label_to_obs_act
        self.custom_functions += [custom_function]

    def _convert_pattern_to_history_graph(self, pattern: history_pattern) -> history_graph:
        return

    def _convert_rules_to_history_graph(self, rules: Dict[Tuple[history, observation], List[action]]) -> history_graph:
        return

    def next_actions(self, history: history, observation: observation) -> List[action]:
        possible_actions = []
        for history_graph in self.history_graphs:
            possible_actions = [
                value for value in possible_actions if value in history_graph.next_actions(history, observation)]
        for custom_function in self.custom_functions:
            possible_actions = [
                value for value in possible_actions if value in custom_function(history, observation)]
        return possible_actions


class history_subset_factory:

    def __init__(self) -> None:
        pass

    def new(self) -> 'history_subset_factory':
        self.hs = history_subset()
        return self

    def add_pattern(self, pattern: history_pattern) -> 'history_subset_factory':
        self.hs.add_pattern(pattern)
        return self

    def add_rules(self, rules: Dict[Tuple[history, observation], List[action]]) -> 'history_subset_factory':
        self.hs.add_rules(rules)
        return self

    def add_history(self, history: history) -> 'history_subset_factory':
        self.hs.add_history(history)
        return self

    def add_custom_function(self, custom_function: Callable[[history, observation], List[action]]) -> 'history_subset_factory':
        self.hs.add_custom_function(custom_function)
        return self

    def create(self) -> history_subset:
        return self.hs


hs_factory = history_subset_factory()

if __name__ == '__main__':

    # hs_factory.new().add_custom_function(lambda obs: "act1" if obs == "obs1" else "act0", {
    #     "act1": 0, "obs1": [0, 0, 1], "act0": 0}).add_rules({}).add_history({}).add_pattern("").create()
