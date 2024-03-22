import functools

import gymnasium
import numpy as np
import random
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

MAX_STEP = 100

# def env(render_mode=None):
#     """
#     The env function often wraps the environment in wrappers by default.
#     You can find full documentation for these methods
#     elsewhere in the developer documentation.
#     """
#     internal_render_mode = render_mode if render_mode != "ansi" else "human"
#     env = MovingCompany(render_mode=internal_render_mode)
#     # This wrapper is only for environments which print results to the terminal
#     if render_mode == "ansi":
#         env = wrappers.CaptureStdoutWrapper(env)
#     # this wrapper helps error handling for discrete action spaces
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     # Provides a wide vareity of helpful user errors
#     # Strongly recommended
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env


class MovingCompany(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "moving_company_v0",
        "render_modes": ["human", "rgb_array"]
    }

    def __init__(self, size: int = 6, seed: int = 42, render_mode = None):
        """The init method takes in environment arguments.
        The environment is a sizexsize grid representing two towers
        separated by distance equal to their height.
        3 agents are spawned randomly in the towers or in the space
        seperating the two towers.
        A package is located at the top of the first tower.
        Goals: Agents have to bring it to the top of the second tower
        the fastest way as possible.

        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """

        self.possible_agents = [f"agent_{i}" for i in range(3)]
        self.directions = [(-1,0), (1,0), (0,-1), (0,1)]
        self.size = size
        self._seed = seed
        self._best_reward = 0
        self.init_grid_environment(seed)
        self.render_mode = render_mode

    def init_grid_environment(self, seed: int):
        self.grid = np.ones((self.size, self.size))
        for i in range(0, self.size):
            for j in range(0, self.size):
                if i == 0 or j == 0 or i == self.size - 1 or j == self.size - 1 \
                or (i > 1 and i < self.size - 2 and j > 1 and j < self.size - 2) \
                    or (i == 1 and  1 < j and j < self.size -2):
                    self.grid[i][j] = 0

        self.grid[1][1] = 5 # Setting the package in initial position
        self.grid[1][self.size - 2] = 4
        self.grid[self.size - 2][1] = 4
        self.grid[self.size - 2][self.size - 2] = 4

        agents_counter = len(self.possible_agents)
        agent_condition_positions = [(None,1), (self.size-2, None), (None, self.size-2)]
        self.agents_position = {agent: (None, None) for agent in self.possible_agents}
        while(agents_counter > 0):
            for i in range(1, self.size-1):
                for j in range(1, self.size-1):
                    if self.grid[i][j] == 1:
                        random.seed(seed)
                        if(random.random() > 0.5):
                            if agents_counter > 0:
                                ic, jc = agent_condition_positions[-1]
                                if (ic is not None and i == ic) or (jc is not None and j == jc):
                                    self.agents_position[f"agent_{agents_counter-1}"] = (i, j)
                                    self.grid[i][j] = 2
                                    agents_counter -= 1
                                    agent_condition_positions.pop()
                                    continue
    
    def apply_action(self, agent_name: str, action: int):
        agent_position = self.agents_position[agent_name]

        if action == 0:
            return
        
        action -= 1

        if 0 <= action and action <= 3:
            direction = self.directions[action]
            targeted_cell_pos = (agent_position[0] + direction[0],agent_position[1] + direction[1])
            # move up, move down, move left, move right
            if self.grid[targeted_cell_pos] == 1:
                agent_cell = self.grid[agent_position]
                self.grid[agent_position] = 1
                self.grid[targeted_cell_pos] = agent_cell
                self.agents_position[agent_name] = targeted_cell_pos
        
        else:
            cross_surrouding_cells = [self.grid[agent_position[0]+direction[0]][agent_position[1]+direction[1]] for direction in self.directions]
            
            # take package
            if action == 4:
                if 5 in cross_surrouding_cells:
                    dir = self.directions[cross_surrouding_cells.index(5)]
                    package_cell_pos = (agent_position[0] + dir[0], agent_position[1] + dir[1])
                    self.grid[package_cell_pos] = 4
                    self.grid[agent_position] = 3

            # drop package
            if action == 5:
                if 4 in cross_surrouding_cells and self.grid[agent_position] == 3:
                    dir = self.directions[cross_surrouding_cells.index(4)]
                    dropzone_cell_pos = (agent_position[0] + dir[0], agent_position[1] + dir[1])
                    self.grid[agent_position] = 2
                    self.grid[dropzone_cell_pos] = 5

    def generate_action_masks(self, agent_name: str):

        action_mask = np.ones(self.action_space(agent_name).n, dtype=np.int8)

        for action in range(3):
            agent_position = self.agents_position[agent_name]

            direction = self.directions[action]
            targeted_cell_pos = (agent_position[0] + direction[0],agent_position[1] + direction[1])

            # move up, move down, move left, move right
            if self.grid[targeted_cell_pos] != 1:
                action_mask[action] = 0
        
        return action_mask
    
    def check_terminated(self) -> bool:
        return self.grid[1][-2] == 5

    def compute_reward(self) -> int:

        package_pos = None
        for i in range(0,self.size):
            for j in range(0,self.size):
                print("ppp: ",self.grid[i][j])
                if self.grid[i][j] in [3,5]:
                    package_pos = (i,j)
                    break
            if package_pos is not None:
                break

        print("mmm ", package_pos)

        best_trajectory = []
        for i in range(1, self.size - 1):
            best_trajectory += [(1, i)]
        for j in range(2, self.size - 1):
            best_trajectory += [(self.size - 2, j)]
        for i in range(1, self.size - 2):
            best_trajectory += [(self.size - 2 - j, self.size - 2)]

        progress_counter = 0
        for i, pos in enumerate(best_trajectory):
            print(package_pos, pos)
            if package_pos == pos:
                break
            progress_counter += 1

        print("=================")

        progress_difference = progress_counter - self._best_reward


        if(self._best_reward < progress_counter):
            self._best_reward = progress_counter

        return progress_difference

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # An agent sees the neighboring cells (3X3 grid):
        # [ ][ ][ ]
        # [ ][X][ ]
        # [ ][ ][ ]
        # Each cell has 5 possible states: Wall (0), Empty (1), Agent (2), Agent+Package (3), EmptyPackageZone (4), NonEmptyPackageZone (5)
        return MultiDiscrete([5] * 3**2, seed=self._seed)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # An has 6 actions: nothing (0), move up (1), move down (2), move left (3), move right (4), take package (5), drop package (6)
        return Discrete(7, seed=self._seed)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.

        Examples of return
        [0][0][0][0][0][0]
        [0][5][0][0][4][0]
        [0][1][0][0][2][0]
        [0][2][0][0][1][0]
        [0][4][1][2][4][0]
        [0][0][0][0][0][0]        
        """

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        print(self.grid)

        if self.render_mode == "human":
            pass # Display pyGame window

        if self.render_mode == "rgba_array":
            pass # Generate an image

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        agent_pos = self.agents_position[agent]
        observation = [0] * (3**2)
        for i, di in enumerate([-1,0,1]):
            for j, dj in enumerate([-1,0,1]):
                observation[i * 3 + j] = self.grid[agent_pos[0]+di][agent_pos[1]+dj]
        return np.array(observation)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        self.init_grid_environment(seed=seed if seed is not None else self._seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_step = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self.apply_action(agent, action)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            common_reward = self.compute_reward()
            # rewards for all agents are placed in the .rewards dictionary
            for ag in self.agents:
                self.rewards[ag] = common_reward

            self.num_step += 1
            if self.num_step > MAX_STEP:
                for ag in self.agents:
                    self.truncations[ag] = True
            
            if self.check_terminated():
                for ag in self.agents:
                    self.terminations[ag] = True

        # observe the current state
        for ag in self.agents:
            self.observations[ag] = {
                "observation": self.observe(ag),
                "action_mask": self.generate_action_masks(ag)
            }

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

if __name__ == '__main__':
    
    env = MovingCompany(size=10, render_mode="rgb_array")

    env.reset(seed=42)

    env.render()

    observation, reward, termination, truncation, info = env.last()
    print(reward)

    env.step(5)
    env.step(0)
    env.step(0)

    observation, reward, termination, truncation, info = env.last()
    print(reward)

    env.step(2)
    env.step(0)
    env.step(2)

    env.step(2)
    env.step(0)
    env.step(2)

    env.step(2)
    env.step(0)
    env.step(2)

    env.step(2)
    env.step(0)
    env.step(2)

    env.step(2)
    env.step(0)
    env.step(2)

    env.step(6)
    env.step(0)
    env.step(0)

    #

    env.step(0)
    env.step(5)
    env.step(0)

    env.step(0)
    env.step(4)
    env.step(0)

    env.step(0)
    env.step(4)
    env.step(0)

    env.step(0)
    env.step(4)
    env.step(0)

    env.step(0)
    env.step(4)
    env.step(0)

    env.step(0)
    env.step(4)
    env.step(0)

    env.step(0)
    env.step(6)
    env.step(0)

    env.step(0)
    env.step(0)
    env.step(5)

    env.step(0)
    env.step(0)
    env.step(1)

    env.step(0)
    env.step(0)
    env.step(1)

    env.step(0)
    env.step(0)
    env.step(1)

    env.step(0)
    env.step(0)
    env.step(1)

    env.step(0)
    env.step(0)
    env.step(1)

    env.step(0)
    env.step(0)
    env.step(6)

    env.render()

    observation, reward, termination, truncation, info = env.last()
    print(reward)

    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         action = None
    #     else:
    #         print("observation: ", observation["observation"], " mask: ", observation["mask"])
    #         action = env.action_space(agent).sample(mask=observation["mask"])

    #     env.step(action)
    # env.close()
