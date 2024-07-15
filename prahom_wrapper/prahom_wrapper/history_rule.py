from typing import Dict, List, Tuple
from utils import label, history, history_pattern_str, history_str
from history_pattern import history_pattern


class history_rules:

    def __init__(self) -> None:
        self.rules: Dict[Tuple[str, label], List[label]] = {}

    def add_rule(self, history_pattern_string: str, observation: label, actions: List[label]) -> None:
        self.rules[(history_pattern_string, observation)] = actions

    def get_actions(self, history: history_str, observation_label: label) -> List[label]:
        actions = [actions for hist_obs, actions in self.rules.items() if history_pattern(
            hist_obs[0]).match(history)[0] and hist_obs[1] == observation_label]
        return actions[0] if len(actions) > 0 else None


if __name__ == '__main__':

    hr = history_rules()
    hr.add_rule("[o1,a1,[#any](0,*),o3](1,1)", "o4", ["a1", "a2", "a3"])
    print(hr.get_actions("o1,a1,kk,k,k,o3", "o4"))
