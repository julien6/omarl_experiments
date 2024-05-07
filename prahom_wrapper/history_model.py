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
from organizational_model import cardinality, os_encoder
from PIL import Image
from pattern_utils import parse_str_sequence, sequence
from utils import draw_networkx_edge_labels

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


class occurrence_number(int):
    pass


class history(List[Union[observation_label, action_label]]):
    pass


class pattern_histories:
    def __init__(self, string_pattern: str) -> None:
        self.pattern = parse_str_sequence(string_pattern)


class occurence_data:
    """A utilitary class to save data about how the transition was played and
    how to play it according to patterns.
    """

    def __init__(self, ordinal_crossing: List[int] = [], global_cardinality: cardinality = cardinality(1, 1),
                 cardinalities: Dict[int, cardinality] = {}, global_priority: int = 0,
                 priorities: Dict[int, int] = {}):
        # used for exhaustive description (e.g from given histories)
        self.ordinal_crossing: List[int] = ordinal_crossing

        # used for short way description (e.g from patterns)
        # indicates the expected cardinality for all cycles
        self.global_cardinality = global_cardinality
        # indicates the expected cardinality for given cycles
        self.cardinalities = cardinalities
        # indicates the expected priority for all cycles
        self.global_priority = global_priority
        self.priorities = priorities  # indicates the expected priority for given cycles

    def compute_pattern_from_data(self) -> None:
        self.global_cardinality = cardinality(
            len(self.ordinal_crossing), len(self.ordinal_crossing))
        self.cardinalities = {
            cycle_num: cardinality(1, 1) for cycle_num, crossing_num
            in enumerate(self.ordinal_crossing)
        }
        self.priorities = {cycle_num: crossing_num for cycle_num, crossing_num
                           in enumerate(self.ordinal_crossing)
                           }

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=0, cls=os_encoder)

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=0, cls=os_encoder)


class indexed_occurences(Dict[int, occurence_data]):

    def __str__(self) -> str:
        return str({hist_num: occ.__str__() for hist_num, occ in self.items()})

    def __repr__(self) -> str:
        return str({hist_num: occ.__str__() for hist_num, occ in self.items()})


class histories:

    # each observation: Any is associated with a shortcut: str
    obs_label_to_obj: Dict[observation_label, Any]
    # each action: Any is associated with a shortcut: str
    act_label_to_obj: Dict[action_label, Any]

    observation_to_actions: Dict[observation_label,
                                 Dict[action_label, indexed_occurences]]

    action_to_observations: Dict[action_label,
                                 Dict[observation_label, indexed_occurences]]

    root_observations: Dict[int, observation_label]

    history_number: int

    def __init__(self) -> None:
        self.histories = {}
        self.obs_label_to_obj = {}  # each observation: Any is associated with a shortcut: str
        self.act_label_to_obj = {}  # each action: Any is associated with a shortcut: str
        self.observation_to_actions = {}
        self.action_to_observations = {}
        self.root_observations = {}
        self.history_number = 0

    def add_histories(self, histories: List[List[Union[observation_label, action_label]]]):
        for history in histories:
            self.add_history(history)

    def add_history(self, sequence: List[Union[observation_label, action_label]]):

        i = 0
        while (i < len(sequence) - 1):

            # processing the action -> observations

            obs_label = (sequence[i] if sequence[i] is not None else f"#obs_{i}") if i < len(
                sequence) else f"#obs_{i}"
            i += 1

            act_label = (sequence[i] if sequence[i] is not None else f"#act_{i}") if i < len(
                sequence) else f"#act_{i}"

            if obs_label not in self.observation_to_actions.keys():
                self.observation_to_actions[obs_label] = {}

            if act_label not in self.observation_to_actions[obs_label].keys():
                self.observation_to_actions[obs_label][act_label] = {}

            if self.history_number not in self.observation_to_actions[obs_label][act_label].keys():
                self.observation_to_actions[obs_label][act_label][self.history_number] = occurence_data(
                    ordinal_crossing=[])

            self.observation_to_actions[obs_label][act_label][self.history_number].ordinal_crossing += [i]
            self.observation_to_actions[obs_label][act_label][self.history_number].compute_pattern_from_data(
            )

            # processing the action -> observations

            i += 1
            next_obs_label = (sequence[i] if sequence[i] is not None else f"#any_obs_{i}") if i < len(
                sequence) else f"#any_obs_{i}"

            if act_label not in self.action_to_observations.keys():
                self.action_to_observations[act_label] = {}

            if next_obs_label not in self.action_to_observations[act_label].keys():
                self.action_to_observations[act_label][next_obs_label] = {}

            if self.history_number not in self.action_to_observations[act_label][next_obs_label].keys():
                self.action_to_observations[act_label][next_obs_label][self.history_number] = occurence_data(
                    ordinal_crossing=[])

            self.action_to_observations[act_label][next_obs_label][self.history_number].ordinal_crossing += [i]
            self.action_to_observations[act_label][next_obs_label][self.history_number].compute_pattern_from_data()

        self.root_observations[self.history_number] = sequence[0]
        self.history_number += 1

    def add_pattern(self, pattern: Union[str, pattern_histories]):

        if type(pattern) == str:
            pattern = pattern_histories(pattern)

        seq = pattern.pattern

        self.first_obs = None

        def add_pattern_aux(sequence_pattern: sequence, sequence_index: int, cardinalities: List[cardinality]):
            data = sequence_pattern.data
            if type(data[0]) == sequence:
                cardinalities = [sequence_pattern.cardinality] + cardinalities
                for seq_index, seq in enumerate(data):
                    add_pattern_aux(seq, seq_index, cardinalities)

            elif type(data[0]) == str:
                i = 0
                while (i < len(data)):

                    # processing obs -> act

                    obs_label = data[i]

                    if (self.first_obs is None):
                        self.first_obs = obs_label

                    i += 1

                    if (i == len(data)):
                        break

                    act_label = data[i]

                    if obs_label not in self.observation_to_actions.keys():
                        self.observation_to_actions[obs_label] = {}

                    if act_label not in self.observation_to_actions[obs_label].keys():
                        self.observation_to_actions[obs_label][act_label] = {}

                    if self.history_number not in self.observation_to_actions[obs_label][act_label].keys():
                        self.observation_to_actions[obs_label][act_label][self.history_number] = occurence_data(
                            cardinalities=[])

                    if self.observation_to_actions[obs_label][act_label][self.history_number] is not None:
                        card = [sequence_pattern.cardinality] + cardinalities
                        card = {i: c for i, c in enumerate(card)}

                    # processing act -> next obs

                    i += 1
                    next_obs_label = data[i]

                    if act_label not in self.action_to_observations.keys():
                        self.action_to_observations[act_label] = {}

                    if next_obs_label not in self.action_to_observations[act_label].keys():
                        self.action_to_observations[act_label][next_obs_label] = {
                            self.history_number: occurence_data(
                                cardinalities=card)
                        }

                    if self.history_number not in self.action_to_observations[act_label][next_obs_label].keys():
                        self.action_to_observations[act_label][next_obs_label][self.history_number] = occurence_data(
                            cardinalities=card)

        self.history_number += 1
        # self.compute_father_to_son_relations()
        # self.compute_root_observations()
        self.root_observations += [self.first_obs]

    def next_actions(self, observation: observation_label) -> Dict[action_label, indexed_occurences]:
        return self.observation_to_actions.get(observation, None)

    def next_observations(self, action: action_label) -> Dict[observation_label, indexed_occurences]:
        return self.action_to_observations.get(action, None)

    def generate_graph_plot(self, show: bool = False, render_rgba: bool = False, save: bool = False, transition_data: List[str] = ['global_cardinality']):

        # Create a directed graph object
        G = nx.DiGraph()

        oriented_edges = {}

        def str_occ_info(occ_data: occurence_data, data_to_display: List[str]) -> str:
            res = '{'

            for hist_num, occ_data in occ_data.items():
                res += str(hist_num) + ':{'
                for data_name, data_value in occ_data.__dict__.items():
                    if data_name in data_to_display:
                        res += f'{data_name}:{data_value.__str__()},'
                res = res[:-1] + '},\n'
            return res[:-2] + '}'

        def walk_graph(root_obs_label: observation_label):
            act_occ = self.observation_to_actions.get(root_obs_label, None)
            if act_occ is None:
                return
            for act_label, act_occurences in act_occ.items():
                G.add_edge(root_obs_label, act_label)
                if (root_obs_label, act_label) in oriented_edges.keys():
                    return
                oriented_edges[(root_obs_label, act_label)
                               ] = str(str_occ_info(act_occurences, transition_data))
                obs_occ = self.action_to_observations.get(act_label)
                for obs_label, obs_occurences in obs_occ.items():
                    G.add_edge(act_label, obs_label)
                    oriented_edges[(act_label, obs_label)
                                   ] = str(str_occ_info(obs_occurences, transition_data))
                    walk_graph(obs_label)

        # Add directed edges (source set)
        for hist_num, root_obs_label in self.root_observations.items():
            walk_graph(root_obs_label)

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

    def walk_with_history(self, history: history):

        def get_observations(act: Union[action_label, None]):
            if act is None:
                return self.root_observations
            return list(self.action_to_observations[act])

        def get_actions(obs: Union[observation_label, None]):
            return list(self.observation_to_actions[obs])

        last_obs = None
        last_act = None
        i = 0
        while (i < len(history)):

            curr_obs = history[i]
            if not curr_obs in get_observations(last_act):
                return False
            i += 1
            last_obs = curr_obs

            if (i == len(history)):
                break

            curr_act = history[i]

            if not curr_act in get_actions(last_obs):
                return False
            i += 1
            last_act = curr_act
        return True


if __name__ == '__main__':

    # histories_factory._instance.new_histories().where_all_match()

    hg = histories()

    # adding sequences to mimic a "(Any_obs, Any_act, Any_obs){0,10}"
    # hg.add_history(["o21", "a21", "o31", "a31", "o21"], {1: (0, 10)})
    # hg.add_history(["o21", "a22", "o2"], {1: (0, 10)})

    # # adding a sequence
    # hg.add_history(["o1", "a1", "o2", "a3", "o1",
    #                 "a1", "o2", "a3", "o1", "a4", "o4"])

    # hg.add_history(["o1", "a1", "o2", "a3", "o1", "a4", "o4"])

    # hg.add_history(["o0", "a0", "o1", "a1", "o1",
    #                 "a1", "o1", "a1", "o2", "a2"])

    hg = histories()
    hg.add_history(["o0", "a0", "o1", "a1", "o1", "a1", "o1", "a1", "o2", "a2"])

    # hg.add_pattern('[[o0,a0,o1,a0,a1](1,1),[o1,a1,o2,a2,o1](1,1),[o0,a0,o1,a0,a1](1,1)](1,2)')

    # hg.add_pattern('[o0,a0,o1](1,1)')

    # hg.add_pattern('[o0,a0,o1,a1,o0,a0,o1](1,1)')

    # str_pattern = "[o3|o4](1,1)"  

    # ph = pattern_histories(str_pattern)

    # print(ph.pattern)

    graph_plot = hg.generate_graph_plot(
        show=True, transition_data=["global_cardinality","priority","cardinalities", "priorities"])

    # hg.add_history(["o0", "a0", "o1", "a1", "o1", "a1", "o1", "a1",
    #                 "o2", "a0", "o1", "a1", "o1", "a1", "o1", "a1", "o2", "a2"])

    # # hg.add_history(["o61", "a61", "o31"], {1: (1, 1)})
    # # hg.add_history(["o71", "a71", "o31"], {1: (1, 1)})

    # hg.generate_graph_plot(show=True)
    # print(hg.walk_with_history(["o21", "a22", "o2", "a23", "o3"]))
