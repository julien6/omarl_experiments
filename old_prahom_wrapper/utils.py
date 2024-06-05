# def compute_father_to_son_relations():
#     observation_to_son: Dict[observation_label,
#                                     Dict[observation_label, bool]] = {}
#     for obs1 in list(observation_to_actions.keys()):
#         observation_to_son.setdefault(obs1, {})
#         act_occ = observation_to_actions[obs1]
#         for act_label, obs_to_act_occ in act_occ.items():
#             for obs2, act_to_obs_occ in action_to_observations[act_label].items():
#                 min1 = 1
#                 min2 = 1
#                 if len(obs_to_act_occ) > 0 and len(act_to_obs_occ) > 0:
#                     min1 = min([int(x)
#                                 for x in list(obs_to_act_occ.keys())])
#                     min2 = min([int(x)
#                                 for x in list(act_to_obs_occ.keys())])
#                 observation_to_son[obs1][obs2] = min(min1, min2) > 0

#     observation_to_father: Dict[observation_label,
#                                         Dict[observation_label, bool]] = {}

#     for obs_father, obs_son_and_mandatory in observation_to_son.items():
#         for obs_son, mandatory in obs_son_and_mandatory.items():
#             observation_to_father.setdefault(obs_son, {})
#             observation_to_father[obs_son][obs_father] = mandatory

#     return observation_to_son, observation_to_father

# def compute_root_observations():

#     def is_own_nth_father(obs_root: observation_label):
#         return is_own_nth_father_aux(copy.copy(obs_root), copy.copy(obs_root), explored_fathers=[])

#     def is_own_nth_father_aux(curr_obs: observation_label, obs_root: observation_label, explored_fathers: List[observation_label] = []):
#         obs_and_mandatory_fathers = observation_to_father.get(
#             curr_obs, None)

#         if (obs_and_mandatory_fathers is None):
#             return False
#         else:
#             res = False
#             for obs_father, mandatory in obs_and_mandatory_fathers.items():
#                 # print("father of ", curr_obs, " -> ",obs_father)
#                 if obs_father in explored_fathers:
#                     break
#                 if obs_father == obs_root:
#                     # print("cycled! ", explored_fathers)
#                     dont_have_extra_father = True
#                     for explored_father in explored_fathers:
#                         if len(observation_to_father[explored_father].keys()) > 1:
#                             dont_have_extra_father = False
#                             break
#                     if (dont_have_extra_father):
#                         res = True
#                     break
#                 res = res or is_own_nth_father_aux(
#                     obs_father, obs_root, explored_fathers + [obs_father])
#             return res

#     root_obs = []

#     for obs in list(observation_to_actions.keys()):
#         obs_fathers = observation_to_father.get(obs, None)
#         if obs_fathers is None:
#             root_obs += [obs]
#         elif is_own_nth_father(obs) and len(observation_to_son[obs].keys()) >= 2:
#             for son_observation in observation_to_son[obs]:
#                 # if one of the son observations is not in a cycle
#                 if not is_own_nth_father(son_observation):
#                     root_obs += [obs]
#                     break

#     root_observations = root_obs



def draw_networkx_edge_labels(
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

