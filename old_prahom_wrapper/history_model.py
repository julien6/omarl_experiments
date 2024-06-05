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
from prahom_wrapper.prahom_wrapper.organizational_model import cardinality, os_encoder
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


class occurrence_number(int):
    pass


class history(List[Union[observation_label, action_label]]):
    pass


class joint_history(Dict[str, List[Union[observation_label, action_label]]]):
    pass


class pattern_histories(object):
    pass


class occurence_data:
    """A utilitary class to save data about how the transition was played and
    how to play it according to patterns.
    """

    def __init__(self, ord_num_to_card: Dict[int, cardinality] = {}):
        # used for exhaustive description (e.g from given histories)
        self.ord_num_to_card = ord_num_to_card

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

    label_to_label: Dict[Union[observation_label, action_label],
                         Dict[Union[observation_label, action_label], indexed_occurences]]

    root_observations: Dict[int, observation_label]

    history_number: int

    def __init__(self) -> None:
        self.histories = {}
        self.obs_label_to_obj = {}  # each observation: Any is associated with a shortcut: str
        self.act_label_to_obj = {}  # each action: Any is associated with a shortcut: str
        self.label_to_label = {}
        self.root_observations = {}
        self.history_number = 0

    def add_histories(self, histories: List[List[Union[observation_label, action_label]]]):
        for history in histories:
            self.add_history(history)

    def add_history(self, history: List[Union[observation_label, action_label]]):

        i = 0
        while (i < len(history) - 1):

            # processing the action -> observations

            obs_label = (history[i] if history[i] is not None else f"#obs_{i}") if i < len(
                history) else f"#obs_{i}"
            i += 1

            if (i == len(history)):
                break

            act_label = (history[i] if history[i] is not None else f"#act_{i}") if i < len(
                history) else f"#act_{i}"

            if obs_label not in self.label_to_label.keys():
                self.label_to_label[obs_label] = {}

            if act_label not in self.label_to_label[obs_label].keys():
                self.label_to_label[obs_label][act_label] = {}

            if self.history_number not in self.label_to_label[obs_label][act_label].keys():
                self.label_to_label[obs_label][act_label][self.history_number] = occurence_data(
                    ord_num_to_card={})

            self.label_to_label[obs_label][act_label][self.history_number].ord_num_to_card[i-1] = cardinality(
                1, 1)

        if self.history_number not in self.root_observations:
            self.root_observations[self.history_number] = []
        self.root_observations[self.history_number] += [history[0]]
        self.history_number += 1

    def add_pattern(self, pattern: str):

        if type(pattern) == str:
            pattern = eval_str_history_pattern(pattern)

        def all_label(labels: Any):
            for label in labels:
                if type(label) == list:
                    for l in label:
                        if type(l) != str:
                            return False
                elif type(label) != str:
                    return False
            return True

        self.ordinal_num = 0
        self.first_obs = None
        self.is_last = False
        self.last_label = None
        self.seq_start_label = None
        self.seq_end_label = None
        self.changed = False

        def add_pattern_aux(pattern: Any, cardinalities: List):

            data = pattern[0]
            card = pattern[1]

            if not all_label(data):
                seq_start = None
                seq_end = None
                for seq_hist in data:
                    start, seq_end = add_pattern_aux(
                        seq_hist, [card]+cardinalities)
                    if seq_start is None:
                        seq_start = start

                if self.label_to_label.get(seq_end, None) == None:
                    self.label_to_label[seq_end] = {}
                if self.label_to_label[seq_end].get(seq_start, None) == None:
                    self.label_to_label[seq_end][seq_start] = {}
                if self.label_to_label[seq_end][seq_start].get(self.history_number, None) == None:
                    self.label_to_label[seq_end][seq_start][self.history_number] = occurence_data(
                        ord_num_to_card={})
                card_seq = [(0 if (int(c) - 1) < 0 else (int(c) - 1)) if c.isnumeric()
                            else "*" for c in pattern[1]]
                self.label_to_label[seq_end][seq_start][self.history_number]\
                    .ord_num_to_card[self.ordinal_num] = cardinality(card_seq[0], card_seq[1])
                self.ordinal_num += 1

            else:
                self.is_last = False
                self.changed = False
                i = 0
                while i < len(data):

                    if i == 0:
                        self.seq_start_label = data[0]
                        self.seq_end_label = data[-1]

                    if i == 0 and self.last_label is not None:
                        data = [self.last_label] + data
                        self.changed = True

                    last_label = data[i]
                    if self.ordinal_num == 0:
                        self.first_obs = copy.copy(last_label)

                    if (i == len(data) - 1):
                        card = [(0 if (int(c) - 1) < 0 else (int(c) - 1)) if c.isnumeric()
                                else "*" for c in pattern[1]]
                        first_label = data[1] if self.changed else data[0]
                        if (len(data) > 1) and ((card[0] > 0 and card[1] > 0) or (card[1] == "*")):
                            data += [first_label]
                        self.is_last = True
                        self.last_label = last_label
                    else:
                        card = ("1", "1")
                    i += 1

                    if i == len(data):
                        break

                    next_label = data[i]
                    if self.label_to_label.get(last_label, None) == None:
                        self.label_to_label[last_label] = {}
                    if self.label_to_label[last_label].get(next_label, None) == None:
                        self.label_to_label[last_label][next_label] = {}
                    if self.label_to_label[last_label][next_label].get(self.history_number, None) == None:
                        self.label_to_label[last_label][next_label][self.history_number] = occurence_data(
                            ord_num_to_card={})
                    self.label_to_label[last_label][next_label][self.history_number]\
                        .ord_num_to_card[self.ordinal_num] = cardinality(card[0], card[1])
                    self.ordinal_num += 1
                    if self.is_last:
                        break
                return self.seq_start_label, self.seq_end_label

        add_pattern_aux(pattern, [])

        if self.history_number not in self.root_observations:
            self.root_observations[self.history_number] = []
        self.root_observations[self.history_number] += [self.first_obs]

        self.history_number += 1

    def next_actions(self, observation: observation_label) -> Dict[action_label, indexed_occurences]:
        return self.label_to_label.get(observation, {})

    def next_observations(self, action: action_label) -> Dict[observation_label, indexed_occurences]:
        return self.label_to_label.get(action, {})

    def generate_graph_plot(self, show: bool = False, render_rgba: bool = False, save: bool = False, transition_data: List[str] = ['ord_num_to_card']):

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
            node_occ = self.label_to_label.get(root_obs_label, None)
            if node_occ is None:
                return
            for node_label, occurences in node_occ.items():
                G.add_edge(root_obs_label, node_label)
                if (root_obs_label, node_label) in oriented_edges.keys():
                    return
                oriented_edges[(root_obs_label, node_label)
                               ] = str(str_occ_info(occurences, transition_data))
                walk_graph(node_label)

        # Add directed edges (source set)
        for hist_num, root_obs_labels in self.root_observations.items():
            for root_obs_label in root_obs_labels:
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

        associated_histories = []
        node_label = history[0]

        if node_label not in list(itertools.chain.from_iterable(list(self.root_observations.values()))):
            return associated_histories
        else:
            for i in range(0, len(history) - 1):
                node_label = history[i]
                next_node_label = history[i+1]

                ass_hist = []

                if self.label_to_label.get(node_label, None) is None or self.label_to_label[node_label].get(next_node_label, None) is None:
                    break
                for hist_num, occ in self.label_to_label[node_label][next_node_label].items():
                    if (i) in list(occ.ord_num_to_card.keys()):
                        ass_hist += [hist_num]

                associated_histories += [ass_hist]

        return associated_histories


class joint_histories:
    
    ag_histories: Dict[str, histories]

    def __init__(self, agents: List[str]):
        self.ag_histories = {agent: histories() for agent in agents}
    
    def add_joint_history(self, jt_hist: joint_history):
        for agent in self.ag_histories:
            self.ag_histories[agent].add_history(jt_hist[agent])
        return self

    def add_joint_histories(self, jt_hist: Dict[str, histories]):
        self.ag_histories = jt_hist
        return self

    def add_history(self, jt_hist: history, agent: str):
        for agent in self.ag_histories:
            self.ag_histories[agent].add_history(jt_hist[agent])
        return self

    def add_joint_pattern_history(self, jt_patt_hist: Dict[str, pattern_histories]):
        for agent in self.ag_histories:
            self.ag_histories[agent].add_pattern(jt_patt_hist)
        return self

    def add_pattern_history(self, pattern_hist: pattern_histories, agent: str):
        self.ag_histories[agent].add_pattern(pattern_hist)
        return self

    def get_next_joint_action(self, joint_observation: Dict[str, observation_label]) -> Dict[str, List[action_label]]:
        return {agent: list(self.ag_histories[agent].next_actions(observation_label).keys()) for agent, observation_label in joint_observation.items()}

if __name__ == '__main__':

    hg = histories()

    # hg.add_history(["o0", "a0", "o1", "a1", "o1",
    #                "a1", "o1", "a1", "o2", "a0", "o1", "a1", "o1",
    #                "a1", "o1", "a1", "o2", "a2", "o3"])

    # hg.add_history(["o0", "a0", "o1"])

    # hg.add_history(["o0", "a0", "o1", "a1", "o2", "a3", "o3"])

    # hg.add_history(["0", "1", "0", "1", "2", "3", "2"])
    # hg.add_history(["0", "1", "0", "1", "0", "1", "0"])
    # hg.add_history(["0", "1", "0"])

    # print(hg.walk_with_history(["0", "1", "0", "1", "2"]))

    # hg.add_pattern("[[o0,a0,o2](1,1)[#any_obs,#any_act](1,*)[o5,a6,o7](1,1)](4,4)")

    # hg.add_pattern("[[o0,a0,o1](1,1)[a2,o2](1,1)](7,7)]")

    # hg.add_pattern("[0,1](1,1)")
    # hg.add_pattern("[0,1](3,3)")
    # print(hg.walk_with_history(["0", "1", "0", "1", "0", "1"]))

    # graph_plot = hg.generate_graph_plot(show=True, transition_data=[
    #                                     "ord_num_to_card"])
