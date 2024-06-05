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
from prahom_wrapper.prahom_wrapper.organizational_model import cardinality, organizational_model, os_encoder
from PIL import Image
from prahom_wrapper.pattern_utils import eval_str_history_pattern, parse_str_history_pattern, history_pattern
from prahom_wrapper.history_model import histories, joint_histories
from prahom_wrapper.utils import draw_networkx_edge_labels


class osj_relation:

    def __init__(self, agents: List[str]) -> None:
        self.agents = agents
        self.os_ag_to_joint_histories: Dict[Tuple[str,
                                                  str], joint_histories] = {}

    def link_os(self, os: organizational_model, joint_histories: joint_histories, agents: List[str]) -> 'osj_relation':
        self.os_ag_to_joint_histories[(
            os.convert_to_label(), ",".join(agents))] = joint_histories
        return self

    def link_role(self, role: organizational_model, role_histories: histories) -> 'osj_relation':
        for agent in self.agents:
            jhs = joint_histories(self.agents)
            jt_histories = {ag: None for ag in self.agents}
            jt_histories[agent] = role_histories
            jhs.add_joint_histories(jt_histories)
            self.link_os(role, jhs, agents=[agent])
        return self

    def get_joint_histories(self, os: organizational_model, agents: List[str]) -> joint_histories:
        return self.os_ag_to_joint_histories[(os.convert_to_label(), ",".join(agents))]
