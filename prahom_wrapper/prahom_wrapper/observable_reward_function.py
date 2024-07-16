import itertools
from typing import Callable, List, Tuple
from history_function import history_functions
from history_rule import history_rules
from utils import label, history, history_pattern_str
from history_pattern import history_pattern, history_patterns

MATCH_REWARD = 100
COVERAGE_REWARD = 10

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

        if type(history) == list:
            if len(history) > 0 and type(history[0]) == tuple:
                history = list(
                    set(list(itertools.chain.from_iterable([[l1, l2] for l1, l2 in history]))))
            history = ",".join(history)

        function_actions = self.history_functions.get_actions(
            history, observation_label)
        pattern_actions = self.history_patterns.get_actions(
            history, observation_label)
        rule_actions = self.history_rules.get_actions(
            history, observation_label)

        return list(set(([] if function_actions is None else function_actions)
                        + ([] if pattern_actions is None else pattern_actions)
                        + ([] if rule_actions is None else rule_actions)))

    def reward(self, history: history) -> float:

        if type(history) == list:
            if len(history) > 0 and type(history[0]) == tuple:
                history = list(
                    set(list(itertools.chain.from_iterable([[l1, l2] for l1, l2 in history]))))
            history = ",".join(history)
        if len(history.split(",")) % 2 == 1:
            raise Exception("History should have full (observation, action) couples")

        reward = 0

        for pattern in self.history_patterns.patterns:
            match, matched, coverage, next_seq = pattern.match(history)
            if match:
                return MATCH_REWARD
            else:
                reward += (2*coverage - 1) * COVERAGE_REWARD

        reward -= MATCH_REWARD

        # history = history.split(",")
        # on_build_history = []
        # still_match = True
        # i = 0
        # while i < len(history):
        #     observation = history[i]
        #     expected_actions = self.get_actions(on_build_history, observation)
            
        #     on_build_history += [observation]

        #     if len(expected_actions) == 0:
        #         break

        #     i += 1
            
        #     action = history[i]
        #     on_build_history += [action]
        #     if action not in expected_actions:
        #         still_match = False
        #         break
            
        #     i += 1

        # if not still_match:
        #     reward = -reward 

        return reward


if __name__ == '__main__':

    # hps = history_patterns()
    # hps.add_pattern("[0,1,2,3,4,5,6,7,8,9](1,1)")
    # print(hps.get_actions("0,1,2", "3"))

    opc = observable_reward_function()
    opc.add_pattern("[0,[#any](0,*),[9,#any](1,1)](1,1)")

    print(opc.reward("0,1,5,6,9,14"))
