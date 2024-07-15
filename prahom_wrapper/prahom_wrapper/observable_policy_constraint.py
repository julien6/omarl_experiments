import itertools
from typing import Callable, Dict, List, Tuple
from history_function import history_functions
from history_rule import history_rules
from utils import label, history, history_pattern_str, history_str
from history_pattern import history_pattern, history_patterns


class observable_policy_constraint:

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
        history = list(
            set(list(itertools.chain.from_iterable([[l1, l2] for l1, l2 in history]))))
        function_actions = self.history_functions.get_actions(
            ",".join(history), observation_label)
        pattern_actions = self.history_patterns.get_actions(
            ",".join(history), observation_label)
        rule_actions = self.history_rules.get_actions(
            ",".join(history), observation_label)

        return list(set(([] if function_actions is None else function_actions)
                        + ([] if pattern_actions is None else pattern_actions)
                        + ([] if rule_actions is None else rule_actions)))


if __name__ == '__main__':

    def manual_custom1(history: history, observation_label: label) -> List[label]:
        if "o2" in history and observation_label == "o13":
            return ["a1", "a2"]
        else:
            return ["a0"]

    opc = observable_policy_constraint()

    opc.add_rule("[o1,a1,[#any](0,*),o3](1,1)", "o4", ["a1", "a2", "a3"])
    opc.add_custom_function(manual_custom1)
    opc.add_pattern("[0,1,2,3,[#any](0,*),4,5,6](1,1)")

    # print(opc.get_actions(["o1", "a1", "o2", "o3"], "o13"))
    print(opc.get_actions([("0", "1"), ("2", "3"), ("79", "81")], "4"))
