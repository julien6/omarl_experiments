from typing import Any, Callable, Dict, List, Tuple, Union
from prahom_wrapper.organizational_model import cardinality, organizational_model, os_encoder
from prahom_wrapper.pattern_utils import eval_str_history_pattern, parse_str_history_pattern, history_pattern
from prahom_wrapper.history_model import action_label, histories, joint_histories, observation_label
from prahom_wrapper.utils import draw_networkx_edge_labels


class joint_policy_constraint:

    jt_histories: joint_histories

    def __init__(self, joint_histories_list: List[joint_histories]) -> None:

        self.jt_histories = joint_histories(
            list(joint_histories_list[0].ag_histories.keys()))

        for _jt_histories in joint_histories_list:
            self.jt_histories.add_joint_histories(_jt_histories.ag_histories)

    def next_actions(self, joint_observation: Dict[str, observation_label]) -> Dict[str, List[action_label]]:
        return self.jt_histories.get_next_joint_action(joint_observation)
