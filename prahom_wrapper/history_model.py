from dataclasses import dataclass, field
import dataclasses
from enum import Enum
import itertools
import json
import random
from typing import Any, Callable, Dict, List, Tuple, Union
from organizational_model import cardinality
import networkx as nx
import matplotlib.pyplot as plt
import netgraph

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


class indexed_occurences(Dict[int, Union[int, str]]):
    """A class to describe the number of times an action is chosen and how many times
    over several histories

    Examples:
    {4: 2, 3: 1} means an action has been chosen 4 consecutive times in two histories
    and that action has been chosen 3 consecutive in just one history/sequence.

    {1:10,2:10,3:10,4:10} means an action can be chosen 1 up to 4 consecutive times over 10 
    histories/sequences.

    {0: "*",1:"*",2:"*",3:"*",4:"*"} means an action can be chosen 1 up to 4 consecutive times over
    all histories. It is equivalent to multiplicity {1,4}.
    """
    pass


class histories_graph:

    observation_to_actions: Dict[observation_label,
                                 Dict[action_label, indexed_occurences]]

    action_to_observations: Dict[action_label,
                                 Dict[observation_label, indexed_occurences]]

    root_observations: List[observation_label]

    def __init__(self) -> None:
        self.histories_graph = {}
        self.observation_to_actions = {}
        self.action_to_observations = {}
        self.root_observations = []

    def add_sequence(self, sequence: List[Union[observation_label, action_label]], occurences: Union[indexed_occurences, cardinality]):

        if type(occurences) == cardinality:
            occurences = self._generate_indexed_occurences(occurences)

        i = 0
        while (i < len(sequence) - 1):

            obs_label = (sequence[i] if sequence[i] is not None else f"#any_obs_{i}") if i < len(
                sequence) else f"#any_obs_{i}"
            i += 1

            act_label = (sequence[i] if sequence[i] is not None else f"#any_act_{i}") if i < len(
                sequence) else f"#any_act_{i}"

            if obs_label not in self.observation_to_actions.keys():
                self.observation_to_actions[obs_label] = {}

            if act_label not in self.observation_to_actions[obs_label].keys():
                self.observation_to_actions[obs_label][act_label] = {}

            self.observation_to_actions[obs_label][act_label] = self._add_indexed_occurences(
                self.observation_to_actions[obs_label][act_label], occurences)

            i += 1
            next_obs_label = (sequence[i] if sequence[i] is not None else f"#any_obs_{i}") if i < len(
                sequence) else f"#any_obs_{i}"

            if act_label not in self.action_to_observations.keys():
                self.action_to_observations[act_label] = {}

            if next_obs_label not in self.action_to_observations[act_label].keys():
                self.action_to_observations[act_label][next_obs_label] = {}

            self.action_to_observations[act_label][next_obs_label] = self._add_indexed_occurences(
                self.action_to_observations[act_label][next_obs_label], occurences)

    def _add_indexed_occurences(self, do_src: indexed_occurences, do_to_add: indexed_occurences) -> indexed_occurences:
        for occurrence_number, history_number in do_to_add.items():
            do_src.setdefault(occurrence_number, 0)
            do_src[occurrence_number] = do_src[occurrence_number] + \
                history_number if history_number != "*" else "*"
        return do_src

    def _generate_indexed_occurences(self, card: cardinality) -> indexed_occurences:
        do = {}
        for i in range(card.lower_bound, card.upper_bound + 1):
            do[i] = "*"
        return do

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

    def generate_graph_plot(self, show: bool = False, save: bool = False):

        # Create a directed graph object
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(list(hg.observation_to_actions.keys()) +
                         list(hg.action_to_observations.keys()))

        oriented_edges = {}

        def walk_graph(root_obs_label: observation_label):
            act_occ = hg.observation_to_actions.get(root_obs_label, None)
            if act_occ is None:
                return
            for act_label, act_occurences in act_occ.items():
                G.add_edge(root_obs_label, act_label)
                if (root_obs_label, act_label) in oriented_edges.keys():
                    return
                oriented_edges[(root_obs_label, act_label)
                               ] = str(act_occurences)
                obs_occ = hg.action_to_observations.get(act_label)
                for obs_label, obs_occurences in obs_occ.items():
                    G.add_edge(act_label, obs_label)
                    oriented_edges[(act_label, obs_label)
                                   ] = str(obs_occurences)
                    walk_graph(obs_label)

        # Add directed edges
        walk_graph(list(hg.observation_to_actions.keys())[0])

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


class history(List[Union[observation_label, action_label]]):
    pass


class histories:
    """A histories basic class to represent a subset of histories for a single agent.
    It represents them as a graph where nodes are observations and edges are the action augmented
    with their occurrence number
    """

    def __init__(self, obs_tag_to_obj: Dict[observation_label, Any], act_label_to_obj: Dict[action_label, Any],
                 graph: histories_graph) -> None:
        self.obs_label_to_obj = {}  # each observation: Any is associated with a shortcut: str
        self.act_label_to_obj = {}  # each action: Any is associated with a shortcut: str
        self.histories_graph: histories_graph = {}

    def walk_with_history(self, hist: history):
        i = 0
        last_obs = None
        last_act = None
        while (i < len(hist)):
            last_obs = hist[i]
            i += 1
            next_actions = self.histories_graph.next_actions(last_obs)
            if next_actions is None:  # the observation is not in the history subset
                return (None, None)
            if last_act is not None:
                previous_observations = self.histories_graph.next_observations(act)
                if last_obs not in previous_observations:
                    return (None, last_act)

            act = hist[i] if i < len(hist) else None
            if act is None:
                return (last_obs, None)
            if act not in next_actions.keys():
                return (last_obs, None)
            i+=1
        return last_obs, last_act

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

    def new_histories(self) -> 'histories_factory':
        return self

    def where_all_match(self) -> 'histories_factory':
        return self

    def create(self) -> histories:
        pass


if __name__ == '__main__':

    # histories_factory._instance.new_histories().where_all_match()

    hg = histories_graph()
    # hg.add_sequence([None, None, None], {3: 1})
    # hg.add_sequence([None, None, "o0", "a0", "o1", "a1", "o2", "a2", "o0", None, None], {3: 1})

    hg.add_sequence(["#any_obs_u0", "#any_act_u1", "#any_obs_u0"], {4: 1})
    hg.add_sequence(["#any_obs_u0", None, "o0", "a0", "o1"], {1: 1})
    hg.add_sequence(["o1", "any_act_u2", "#any_obs_u4"], {1: 1})
    hg.add_sequence(["#any_obs_u4", "#any_act_u7", "#any_obs_u4"], {3: 1})
    hg.add_sequence(["#any_obs_u4", "#any_act_u8", "o10"], {1: 1})
    # hg.add_sequence(["any_obs_42",None, "o0", "a0", "o1", "a1", "o2", "a2", "o3"], {1: 1})
    # hg.add_sequence(["o2", "a7", "o1", "a1", "o2"], {3: 1})
    # hg.add_sequence(["o2", "a7", "o8"], {1: 1})

    # print(hg.observation_to_actions)
    # print(hg.action_to_observations)

    hg.generate_graph_plot(show=True)
