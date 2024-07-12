from typing import Dict, List, Tuple
from history_graph import observation
from llm_manager import llm_manager

class actions_manager:

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self.text_id_to_actions: Dict[Tuple[str,
                                                 str], List[observation]] = {}
        self.llm_manager = None
        if self.use_llm:
            self.llm_manager = llm_manager()

    def map_text_description_to_actions(self, text_description: str, actions: List[observation], use_llm: bool = True, actions_identifier: str = None) -> 'actions_manager':
        """One-to-one map a textual description to a set of actions using a regular dict and by default a LLM is prompt-engineered to learn the mapping as well.

        Parameters
        ----------
        text_description : str
            The text describing a set of actions.

        actions: List[str]
            The list of actions to be mapped.

        use_llm: bool (default=True)
            Whether using a LLM to learn how to match actions to a text description in addition to the ones mapped
            with a regular dictionary

        actions_identifier: str (default=None)
            An short-way to identify a unique set of actions

        Returns
        -------
        actions_manager
            The current instance

        Examples
        --------
        >>> 

        See Also
        --------
        None
        """
        self.use_llm = use_llm
        if self.use_llm:
            pass
        else:
            self.text_id_to_actions[(
                actions_identifier, text_description)] = actions
        return self

    def add_actions_from_text_description(self, text_description: str, use_llm: bool = True) -> 'actions_manager':
        """Add a set of actions that match given text description.

        Parameters
        ----------
        text_description : str
            The text describing a set of actions.

        use_llm: bool (default=True)
            Whether using a LLM to find corresponding actions in addition to the ones mapped
            with a regular dictionary

        Returns
        -------
        actions_manager
            The current instance

        Examples
        --------
        >>> 

        See Also
        --------
        None
        """
        self.use_llm = use_llm
        return self

    def set_use_llm(self, use_llm) -> 'actions_manager':
        self.use_llm = use_llm
        return self


def assisted_actions_manager_create(use_llm: bool = True) -> actions_manager:
    """Create an hand-crafted actions_manager by running an interactive assisted process
    consists in running a series of episodes during learning. In each episode, users have to
    describe a visual representation of agents' action for some/all frames.

    Parameters
    ----------
    use_llm: bool (default=True)
        Whether using a LLM to learn how to match actions to a text description in
        addition to the ones mapped with a regular dictionary

    Returns
    -------
    actions_manager
        The current instance

    Examples
    --------
    >>> 

    See Also
    --------
    None
    """
    return actions_manager()


simple_spread_obs_mngr = actions_manager()

simple_world_comm_obs_mngr = actions_manager()
