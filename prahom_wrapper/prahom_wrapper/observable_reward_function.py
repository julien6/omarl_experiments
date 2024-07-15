import itertools
from typing import Callable, List, Tuple
from history_function import history_functions
from history_rule import history_rules
from utils import label, history, history_pattern_str
from history_pattern import history_patterns


class observable_reward_function:

    def __init__(self) -> None:
        self.history_functions = history_functions()
        self.history_patterns = history_patterns()
        self.history_rules = history_rules()

    def add_custom_function(self, function: Callable[[Tuple[history, label]], List[label]]) -> None:
        self.history_functions.add_function(function)

    def add_pattern(self, history_pattern_string: history_pattern_str) -> None:
        self.history_patterns.add_pattern(history_pattern_string)

    def add_rule(self, history_pattern_string: str, observation: label, actions: List[label]) -> None:
        self.history_rules.add_rule(
            history_pattern_string, observation, actions)

    def get_actions(self, history: history, observation_label: label) -> List[label]:
        history = list(set(list(itertools.chain.from_iterable([[l1,l2] for l1,l2 in history]))))
        function_actions = self.history_functions.get_actions(
            ",".join(history), observation_label)
        pattern_actions = self.history_patterns.get_actions(
            ",".join(history), observation_label)
        rule_actions = self.history_rules.get_actions(
            ",".join(history), observation_label)

        return list(set(([] if function_actions is None else function_actions)
                        + ([] if pattern_actions is None else pattern_actions)
                        + ([] if rule_actions is None else rule_actions)))

    def walk_history(self, history: history) -> float:

        for label in history:
