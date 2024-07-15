import importlib
import inspect

import itertools
from typing import Callable, Dict, List, Tuple
from utils import label, history, history_str
from history_pattern import history_pattern


class history_functions:

    def __init__(self):
        self.custom_functions = []

    def add_function(self, function: Callable[[Tuple[history, label]], List[label]]) -> None:

        # source_code = inspect.getsource(function)
        module_name = function.__module__
        self.custom_functions += [{
            'function_name': function.__name__,
            'module_name': module_name  # ,
            # 'source_code': source_code
        }]

    # def add_custom_function_src(self, file_path: str) -> None:
    #     self.custom_functions += [{

    #     }]

    # def add_custom_function_script(self, script: str) -> None:
    #     self.custom_functions += [{

    #     }]

    def load_function(self, data: Dict) -> Callable[[Tuple[history, label]], List[label]]:
        function_name = data['function_name']
        module_name = data['module_name']

        module = importlib.import_module(module_name)

        return getattr(module, function_name)

    def get_actions(self, history: history_str, observation_label: label) -> List[label]:
        actions = []
        for custom_function in self.custom_functions:
            actions += [self.load_function(custom_function)
                        (history.split(","), observation_label)]

        return list(set(list(itertools.chain.from_iterable(actions))))


if __name__ == '__main__':

    def manual_custom1(history: history, observation_label: label) -> List[label]:
        if "o2" in history and observation_label == "o13":
            return ["a1", "a2"]
        else:
            return ["a0"]

    def manual_custom2(history: history, observation_label: label) -> List[label]:
        if "o3" in history:
            return ["a1", "a2"]
        return ["a12", "a14"]

    cf = history_functions()
    cf.add_function(manual_custom1)
    cf.add_function(manual_custom2)
    print(cf.get_actions("o2,a3,o3,a3,o4,a4", "o13"))
