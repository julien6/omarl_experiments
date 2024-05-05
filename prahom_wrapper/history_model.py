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
from organizational_model import cardinality, os_encoder

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

    root_observations: List[observation_label]

    sequence_number: int

    def __init__(self) -> None:
        self.histories = {}
        self.obs_label_to_obj = {}  # each observation: Any is associated with a shortcut: str
        self.act_label_to_obj = {}  # each action: Any is associated with a shortcut: str
        self.observation_to_actions = {}
        self.action_to_observations = {}
        self.root_observations = []
        self.sequence_number = 0

    def add_histories(self, histories: List[List[Union[observation_label, action_label]]]):
        for history in histories:
            self.add_history(history)

    def add_history(self, sequence: List[Union[observation_label, action_label]], occurences: indexed_occurences = None):

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
                self.observation_to_actions[obs_label][act_label] = {
                    self.sequence_number: occurence_data(ordinal_crossing=[])
                }

            if self.sequence_number not in self.observation_to_actions[obs_label][act_label].keys():
                self.observation_to_actions[obs_label][act_label][self.sequence_number] = occurence_data(
                    ordinal_crossing=[])

            self.observation_to_actions[obs_label][act_label][self.sequence_number].ordinal_crossing += [i]
            self.observation_to_actions[obs_label][act_label][self.sequence_number]\
                .compute_pattern_from_data()

            occ_obs_to_act = copy.copy(occurences)
            if occurences is None:
                occ_obs_to_act = {}

            self.observation_to_actions[obs_label][act_label] = self._add_indexed_occurences(
                self.observation_to_actions[obs_label][act_label], occ_obs_to_act)

            # processing the action -> observations

            i += 1
            next_obs_label = (sequence[i] if sequence[i] is not None else f"#any_obs_{i}") if i < len(
                sequence) else f"#any_obs_{i}"

            if act_label not in self.action_to_observations.keys():
                self.action_to_observations[act_label] = {}

            if next_obs_label not in self.action_to_observations[act_label].keys():
                self.action_to_observations[act_label][next_obs_label] = {
                    self.sequence_number: occurence_data(ordinal_crossing=[])
                }

            if self.sequence_number not in self.action_to_observations[act_label][next_obs_label].keys():
                self.action_to_observations[act_label][next_obs_label][self.sequence_number] = occurence_data(
                    ordinal_crossing=[])

            self.action_to_observations[act_label][next_obs_label][self.sequence_number].ordinal_crossing += [i]
            self.action_to_observations[act_label][next_obs_label][self.sequence_number]\
                .compute_pattern_from_data()

            occ_act_to_obs = copy.copy(occurences)
            if occurences is None:
                occ_act_to_obs = {}

            self.action_to_observations[act_label][next_obs_label] = self._add_indexed_occurences(
                self.action_to_observations[act_label][next_obs_label], occ_act_to_obs)

        self.sequence_number += 1
        self.compute_father_to_son_relations()
        self.compute_root_observations()

    def _add_indexed_occurences(self, do_src: indexed_occurences, do_to_add: indexed_occurences) -> indexed_occurences:
        do_src.update(do_to_add)
        return do_src

    def compute_father_to_son_relations(self):
        self.observation_to_son: Dict[observation_label,
                                      Dict[observation_label, bool]] = {}
        for obs1 in list(self.observation_to_actions.keys()):
            self.observation_to_son.setdefault(obs1, {})
            act_occ = self.observation_to_actions[obs1]
            for act_label, obs_to_act_occ in act_occ.items():
                for obs2, act_to_obs_occ in self.action_to_observations[act_label].items():
                    min1 = 1
                    min2 = 1
                    if len(obs_to_act_occ) > 0 and len(act_to_obs_occ) > 0:
                        min1 = min([int(x)
                                   for x in list(obs_to_act_occ.keys())])
                        min2 = min([int(x)
                                   for x in list(act_to_obs_occ.keys())])
                    self.observation_to_son[obs1][obs2] = min(min1, min2) > 0

        self.observation_to_father: Dict[observation_label,
                                         Dict[observation_label, bool]] = {}

        for obs_father, obs_son_and_mandatory in self.observation_to_son.items():
            for obs_son, mandatory in obs_son_and_mandatory.items():
                self.observation_to_father.setdefault(obs_son, {})
                self.observation_to_father[obs_son][obs_father] = mandatory

    def compute_root_observations(self):

        def is_own_nth_father(obs_root: observation_label):
            return is_own_nth_father_aux(copy.copy(obs_root), copy.copy(obs_root), explored_fathers=[])

        def is_own_nth_father_aux(curr_obs: observation_label, obs_root: observation_label, explored_fathers: List[observation_label] = []):
            obs_and_mandatory_fathers = self.observation_to_father.get(
                curr_obs, None)

            if (obs_and_mandatory_fathers is None):
                return False
            else:
                res = False
                for obs_father, mandatory in obs_and_mandatory_fathers.items():
                    # print("father of ", curr_obs, " -> ",obs_father)
                    if obs_father in explored_fathers:
                        break
                    if obs_father == obs_root:
                        # print("cycled! ", explored_fathers)
                        dont_have_extra_father = True
                        for explored_father in explored_fathers:
                            if len(self.observation_to_father[explored_father].keys()) > 1:
                                dont_have_extra_father = False
                                break
                        if (dont_have_extra_father):
                            res = True
                        break
                    res = res or is_own_nth_father_aux(
                        obs_father, obs_root, explored_fathers + [obs_father])
                return res

        root_obs = []

        for obs in list(self.observation_to_actions.keys()):
            obs_fathers = self.observation_to_father.get(obs, None)
            if obs_fathers is None:
                root_obs += [obs]
            elif is_own_nth_father(obs) and len(self.observation_to_son[obs].keys()) >= 2:
                for son_observation in self.observation_to_son[obs]:
                    # if one of the son observations is not in a cycle
                    if not is_own_nth_father(son_observation):
                        root_obs += [obs]
                        break

        self.root_observations = root_obs

    def next_actions(self, observation: observation_label) -> Dict[action_label, indexed_occurences]:
        return self.observation_to_actions.get(observation, None)

    def next_observations(self, action: action_label) -> Dict[observation_label, indexed_occurences]:
        return self.action_to_observations.get(action, None)

    def my_draw_networkx_edge_labels(
        self,
        G,
        pos,
        edge_labels=None,
        label_pos=0.5,
        font_size=10,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        rotate=True,
        clip_on=True,
        rad=0
    ):
        """Draw edge labels.

        Parameters
        ----------
        G : graph
            A networkx graph

        pos : dictionary
            A dictionary with nodes as keys and positions as values.
            Positions should be sequences of length 2.

        edge_labels : dictionary (default={})
            Edge labels in a dictionary of labels keyed by edge two-tuple.
            Only labels for the keys in the dictionary are drawn.

        label_pos : float (default=0.5)
            Position of edge label along edge (0=head, 0.5=center, 1=tail)

        font_size : int (default=10)
            Font size for text labels

        font_color : string (default='k' black)
            Font color string

        font_weight : string (default='normal')
            Font weight

        font_family : string (default='sans-serif')
            Font family

        alpha : float or None (default=None)
            The text transparency

        bbox : Matplotlib bbox, optional
            Specify text box properties (e.g. shape, color etc.) for edge labels.
            Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

        horizontalalignment : string (default='center')
            Horizontal alignment {'center', 'right', 'left'}

        verticalalignment : string (default='center')
            Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

        ax : Matplotlib Axes object, optional
            Draw the graph in the specified Matplotlib axes.

        rotate : bool (deafult=True)
            Rotate edge labels to lie parallel to edges

        clip_on : bool (default=True)
            Turn on clipping of edge labels at axis boundaries

        Returns
        -------
        dict
            `dict` of labels keyed by edge

        Examples
        --------
        >>> G = nx.dodecahedral_graph()
        >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

        Also see the NetworkX drawing examples at
        https://networkx.org/documentation/latest/auto_examples/index.html

        See Also
        --------
        draw
        draw_networkx
        draw_networkx_nodes
        draw_networkx_edges
        draw_networkx_labels
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            ax = plt.gca()
        if edge_labels is None:
            labels = {(u, v): d for u, v, d in G.edges(data=True)}
        else:
            labels = edge_labels
        text_items = {}
        for (n1, n2), label in labels.items():
            (x1, y1) = pos[n1]
            (x2, y2) = pos[n2]
            (x, y) = (
                x1 * label_pos + x2 * (1.0 - label_pos),
                y1 * label_pos + y2 * (1.0 - label_pos),
            )
            pos_1 = ax.transData.transform(np.array(pos[n1]))
            pos_2 = ax.transData.transform(np.array(pos[n2]))
            linear_mid = 0.5*pos_1 + 0.5*pos_2
            d_pos = pos_2 - pos_1
            rotation_matrix = np.array([(0, 1), (-1, 0)])
            ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
            ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
            ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
            bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
            (x, y) = ax.transData.inverted().transform(bezier_mid)

            if rotate:
                # in degrees
                angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = ax.transData.transform_angles(
                    np.array((angle,)), xy.reshape((1, 2))
                )[0]
            else:
                trans_angle = 0.0
            # use default box of white with white border
            if bbox is None:
                bbox = dict(boxstyle="round", ec=(
                    1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
            if not isinstance(label, str):
                label = str(label)  # this makes "1" and 1 labeled the same

            t = ax.text(
                x,
                y,
                label,
                size=font_size,
                color=font_color,
                family=font_family,
                weight=font_weight,
                alpha=alpha,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                rotation=trans_angle,
                transform=ax.transData,
                bbox=bbox,
                zorder=1,
                clip_on=clip_on,
            )
            text_items[(n1, n2)] = t

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        return text_items

    def generate_graph_plot(self, show: bool = False, save: bool = False, transition_data: List[str] = ['global_cardinality']):

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
            act_occ = hg.observation_to_actions.get(root_obs_label, None)
            if act_occ is None:
                return
            for act_label, act_occurences in act_occ.items():
                G.add_edge(root_obs_label, act_label)
                if (root_obs_label, act_label) in oriented_edges.keys():
                    return
                oriented_edges[(root_obs_label, act_label)
                               ] = str(str_occ_info(act_occurences, transition_data))
                obs_occ = hg.action_to_observations.get(act_label)
                for obs_label, obs_occurences in obs_occ.items():
                    G.add_edge(act_label, obs_label)
                    oriented_edges[(act_label, obs_label)
                                   ] = str(str_occ_info(obs_occurences, transition_data))
                    walk_graph(obs_label)

        # # Add nodes
        # G.add_nodes_from(list(hg.observation_to_actions.keys()) +
        #                  list(hg.action_to_observations.keys()))

        # Add directed edges (source set)
        for root_obs_label in self.root_observations:
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
        self.my_draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

        # Show the plot
        if (save):
            fig.savefig(f"{random.randint(1, 100)}.png",
                        bbox_inches='tight', pad_inches=0)
        if (show):
            plt.show()

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


class histories_factory:
    """The basic class
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        pass

    def new_histories(self, obs_tag_to_obj: Dict[observation_label, Any], act_label_to_obj: Dict[action_label, Any],
                      graph: histories = None) -> 'histories_factory':
        self.histories = histories(obs_tag_to_obj, act_label_to_obj, graph)
        return self

    def where_all_match(self) -> 'histories_factory':
        return self

    def create(self) -> histories:
        self.histories.compute_root_observations()
        return self.histories


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

    hg.add_history(["o0", "a0", "o1", "a1", "o1",
                   "a1", "o1", "a1", "o2", "a2"])
    hg.add_histories([["o0", "a0", "o1"], ["o2", "a1", "o3"]])

    # hg.add_history(["o0", "a0", "o1", "a1", "o1", "a1", "o1", "a1",
    #                 "o2", "a0", "o1", "a1", "o1", "a1", "o1", "a1", "o2", "a2"])

    # # hg.add_history(["o61", "a61", "o31"], {1: (1, 1)})
    # # hg.add_history(["o71", "a71", "o31"], {1: (1, 1)})

    hg.generate_graph_plot(show=True)
    # print(hg.walk_with_history(["o21", "a22", "o2", "a23", "o3"]))
