# Organization oriented Multi-Agent Reinforcement Learning

Organization oriented Multi-Agent Reinforcement Learning (OMARL) is a broad topic gathering research that specifically aims at integrating organization model of a Multi-Agent System (MAS) into Multi-Agent Reinforcement Learning (MARL) to improve its explainability and handability for MAS.

In a regular MARL scenario, agents' policies (their rules) are updated via an algorithm until reaching a goal. That process results in a set of trained agents who collaborates to fullfil a goal.

Unfortunately, when getting interested to have an organization point-of-view, the "trained" or "on training" joint-policy gives "learned" rules that could be mixing the individual, social (interactions in neighborhood) or even collective level (interactions with majority).

Even though some works have been pushing forward interest for emergent collective phenomena in MARL, none has considered using an organization model to characterize them (as cooperation schemes, roles, missions...); or even driving the agents' learning through extra organizational specifications.

OMARL's original interest is to have an explicit way to view and act on a set learning agents not only at individual but also social and collective levels.

## Dec-POMDP $\mathcal{M}OISE^+$ OMARL (DMO)

*Dec-POMDP* $\mathcal{M}OISE^+$ *OMARL* (DMO) is first attempt to implement a process that fits OMARL goal as for linking:
 - **Decentralized Partially Observable Markov Decision Process** (Dec-PODMP): a model for using MARL in any formalized scenario; 
 - **$\mathcal{M}OISE^+$**: an organizational model that relies on structural, functional and deontic specifications to describe a MAS organization

We proposed the *Partial Relations with Agent History and Organization Model (PRAHOM)* (PRAHOM), a DMO process that allows:
 - Getting the resulting organizational specifications out of a given joint-policy by linking actions with known organizational specification; 
 - Constraining the agents' training to respect some organizational specifications by restricting available agents' actions at each step

## PRAHOM Gym-wrapper

The PRAHOM process resulted in an algorithm we integrated within the Pettingzoo interface under the form of a Gym wrapper for easy use. That Gym wrapper is to be used to apply ORMARL on environment that requires collaboration and organization among agents.

A typical workflow would consist in:

1) Associating actions with known organizational specifications
2) Defining specifications to respect as extra constraints during training
3) Launching the training over several iterations (or epoch) to get several successful joint-policies
4) Extracting some organizational specifications out of thee joint-policies concerning the emergent organization

## Installation

1) Clone current repo in a safe place
   
2) Create a new clean virtual python environment:

```cd ~; mkdir python-envs; cd python-envs; python -m venv prahom; source prahom/bin/activate; cd ~

```

1) In cloned repo, type:

```python install requirements.txt```

4) Then, type:

```pip install -e .

```

## PRAHOM Gym-wrapper use:

Basic example

```python
from pettingzoo.mpe import simple_world_comm_v3
from omarl_experiments import PrahomWrapper

env = simple_world_comm_v3.parallel_env(render_mode="human")

action_to_specs = ...
training_specs = ...

# wrapping the initial environment to use PRAHOM
env = PrahomWrapper(env, action_to_specs, training_specs, unknown_specs_inference=True, pca_output=True)

observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: policies[] env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

# getting prahom output
env.prahom_render_pca() # print the joint-policies in a PCA for comparison between agents (help to analyze similar agents behaviors)
trained_specs, agent_to_specs = env.prahom_specs()

env.close()

```

We start defining actions with known specifications
Focusing of the "lead adversary", we have:
* "observations": [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities, leader_comm]
* "actions": [say_0, say_1, say_2, say_3] X [no_action, move_left, move_right, move_down, move_up]

```json
action_to_specs = {
    "leadadversary_0": {
        "0": "role='leader';link='(leadadversary_0,adversary_1,auth);...'", //send "move left" to "adversary 0" (agent1)
        ...
    }
    ...
}
```

Then, we define organizational constraints to respect during agents' training.
Here the "lead adversary" has to communicate when it receives low agent velocity but it does not communicate when it is too high.

```json
training_specs = {
    "leadadversary_0": {
        "must": ["(23,41)"], //(obs: agent_0 low speed, communicate up to adversary 0)
        "must_not": ["(14,74)"] //(obs: agent_0 high speed, communicate up to adversary 0)
    }
    ...
}
```

***Note: Learning should converge faster as search space is reduced***

Finally, we can extract the organizational specifications out of the resulting

```python
env.prahom_render_pca() # print the joint-policies in a PCA for comparison
trained_specs, agent_to_specs = env.prahom_specs()
```

Printing the "agent_to_specs"

```json
{
    "groups": {
        "preys": ["agent_0", "agent_1"],
        "predators": ["agent_2", ..., "agent_6"]
    },
    "roles": {
        "agent_0": "prey" // or described by its labels or set of rules if it does not exist as a known role
        ...
    }
    ...
}
```

Printing the "trained_specs"

```json
{
    "structural_specs": {
        "roles": ["predatorleader", "predator"],
        "links": ["link(predatorleader,preadotr,comm)",...],
        "compatibilities": [],
        "subgroups":
    },
    "functional_specs": {
        "goals": ["touchprey", "move_up", "move_down"...],
        "missions": ["swandich_strategy"...],
        "plans": ["touch_prey=(move_up|move_down|move_left|move_right)",],
    },
    "deontic_specs": {
        "permission": ["(predatorleader, sandwich_strategy)",...],
        "obligations": ["(preador, sandwich_strategy)", "(predatorleader, command_mission)"...]
    }
}
```

_______
_______

## General results

<figure>

    <img src="./assets/images/prahom_learning_curve.png"
         alt="PRAHOM Learning curve">
    <figcaption>Learning curve for the NTS, FTS, and PTS cases over 1000 training iterations in the Pistonball environment*</figcaption>

</figure>

<br>

<figure>

    <img src="./assets/images/prahom_pca_analysis.png"
         alt="PRAHOM Learning curve">
    <figcaption>Principal Component Analysis of the trained agents' histories in the Pistonball environment in the NTS</figcaption>

</figure>

## PRAHOM details

PRAHOM is a synthesis of two processes:
 1) Determining $\mathcal{M}OISE^+$ organizational specifications from joint-histories
 2) Constraining possible trained agents' policies according to given $\mathcal{M}OISE^+$ organizational specifications

In both case, it relies on the relations between histories obtained after training and $\mathcal{M}OISE^+$ organizational specifications. Formally, we introduce the relation $rhos$ that associate some specific sub joint-histories to some organizational specifications:

$rhos: \mathcal{P}(H_{joint}) \rightarrow OS$ 

### Inferring organizational specifications

Inferring organizational specifications from joint-histories obtained after training is indeed improving $rhos$

In order to improve $rhos$, we do as follow:
1) Optionally, some relation may already be known.
2) Using the definitions of the organizational specifications regarding histories to determine roles, links, goals for each joint-history; and according to optional known relations in (1)
    - Role: defined by stereotyped behavior
    - Links: defined by impact on other agents whether in the future actions
    - Goals: defined by limited number of similar state reached by agents before achieving the ultimate goal
3) Using the inferred information from each joint-history to infer general organizational specifications such as Compatibilities, Mission plans, Permissions, Obligations...
    - Compatibilities: defined if a same agent can adopt two different roles
    - Mission plans: defined by the way goals are achieved to get the ultimate goal
    - Permissions/Obligations: defined by the way (duration, frequency...) a role is associated to an agent

### Constraining resulting joint-policies

For each step, for each agent:

1) Converting the given organizational specifications (intersected with current agent history) into authorized and forbidden (observation, action) couples
    - Reusing the previously inferred $rhos$ relation
2) MARL update the agent policy from non-constrained observation and action sets
3) Update the history after chosen the next action to play
_______

## Moving Company

![alt text](/assets/images/moving_company_v0.gif)

Moving Company (MCY) is a two-dimensional grid game where mover employees have to bring a package from a cell to a final cell. They are free to move up, left, down, and right in the white cells. They can pick up or drop down the package in the drop zone (yellow cells). The white cells are empty and the grey cells represent walls.
The game ends when the package is dropped in the final cell.
The environment is fully discrete, vectorized. Agents' observations are the 3x3 grid cells surrounding an agent.

_______

## Predator-prey with communication

Source link: https://pettingzoo.farama.org/environments/mpe/simple_world_comm/

![alt text](https://pettingzoo.farama.org/_images/mpe_simple_world_comm.gif)

Simple World Comm is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By default, there is 1 good agent, 3 adversaries and 2 obstacles.

Additionally there is food (small blue balls) that the good agents are rewarded for being near, there are ‘forests’ that hide agents inside from being seen, and there is a ‘leader adversary’ that can see the agents at all times and can communicate with the other adversaries to help coordinate the chase. By default, there are 2 good agents, 3 adversaries, 1 obstacles, 2 foods, and 2 forests.

In particular, the good agents reward, is -5 for every collision with an adversary, -2 x bound by the bound function described in simple_tag, +2 for every collision with a food, and -0.05 x minimum distance to any food. The adversarial agents are rewarded +5 for collisions and -0.1 x minimum distance to a good agent.

_______

## Knights Archers Zombies

Source link: https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/

![alt text](https://pettingzoo.farama.org/_images/butterfly_knights_archers_zombies.gif)

Knights Archers Zombies (KAZ) is a two-dimensional game where zombies walk from the top border of the screen down to the bottom border in unpredictable paths. The agents you control are knights and archers (default 2 knights and 2 archers) that are initially positioned at the bottom border of the screen. Each agent can rotate clockwise or counter-clockwise and move forward or backward. Each agent can also attack to kill zombies. When a knight attacks, it swings a mace in an arc in front of its current heading direction. When an archer attacks, it fires an arrow in a straight line in the direction of the archer’s heading. The game ends when all agents die (collide with a zombie) or a zombie reaches the bottom screen border. A knight is rewarded 1 point when its mace hits and kills a zombie. An archer is rewarded 1 point when one of their arrows hits and kills a zombie. There are two possible observation types for this environment, vectorized and image-based.

_______

## Pistonball

https://pettingzoo.farama.org/environments/butterfly/pistonball/

![alt text](https://pettingzoo.farama.org/_images/butterfly_pistonball.gif)

Pistonball is a simple physics based cooperative game where the goal is to move the ball to the left wall of the game border by activating the vertically moving pistons. Each piston agent’s observation is an RGB image of the two pistons (or the wall) next to the agent and the space above them. Every piston can be acted on in any given time. The action space in discrete mode is 0 to move down, 1 to stay still, and 2 to move up. In continuous mode, the value in the range [-1, 1] is proportional to the amount that the pistons are raised or lowered by. Continuous actions are scaled by a factor of 4, so that in both the discrete and continuous action space, the action 1 will move a piston 4 pixels up, and -1 will move pistons 4 pixels down.

Accordingly, pistons must learn highly coordinated emergent behavior to achieve an optimal policy for the environment. Each agent gets a reward that is a combination of how much the ball moved left overall and how much the ball moved left if it was close to the piston (i.e. movement the piston contributed to). A piston is considered close to the ball if it is directly below any part of the ball. Balancing the ratio between these local and global rewards appears to be critical to learning this environment, and as such is an environment parameter. The local reward applied is 0.5 times the change in the ball’s x-position. Additionally, the global reward is change in x-position divided by the starting position, times 100, plus the time_penalty (default -0.1). For each piston, the reward is local_ratio * local_reward + (1-local_ratio) * global_reward. The local reward is applied to pistons surrounding the ball while the global reward is provided to all pistons.

Pistonball uses the chipmunk physics engine, and are thus the physics are about as realistic as in the game Angry Birds.

_______

## CybORG - Third CAGE Challenge

https://github.com/cage-challenge/cage-challenge-3

![alt text](https://raw.githubusercontent.com/cage-challenge/cage-challenge-3/main/images/scenario-map.png)

"The nation of Florin is conducting reconnaissance on the border with Guilder during a period of tension between the two nations. A set of autonomous aerial drones is used to support soldiers patrolling the border. These drones are designed to form an ad-hoc network, so that any data the soldiers need to transmit to one another can be relayed via the drones as shown in Figure. The drones spread out across the area of interest and aim to maintain sufficient proximity with soldiers in order to enable communications. Guilder is not expected to attempt to destroy the drones, as this would be interpreted as an act of aggression. However, Guilder has experience in conducting cyber operations against Florin, and they may attempt to use their cyber capability to interfere with the mission.

Cyber Threat Intelligence reports indicate that Guilder may have installed hardware Trojans on the drones in the swarm, however the conditions for the activation of the Trojans are unknown. Once activated, these Trojans will deploy a worm that will attempt to compromise other drones in the swarm. This worm may either steal data on Florin’s troop movements, or introduce false information to mislead Florin command.

You are a developer of autonomous defence systems. Following your success defending a Florin munitions factory against attack using autonomous agents (CAGE Challenges 1 and 2), you have been tasked with developing a multi-agent autonomous defence system for the drone support team. The drones are constantly moving, both to maintain the overall network and to track the movements of particular soldiers or vehicles. Communications between any two drones may drop out or be re-established at any time. A centralised approach to cyber defence will be difficult; instead, you will develop a decentralised defence system.

Your primary goals are to defend the drone team such that (a) compromised drones are detected and then isolated or reclaimed, and (b) the flow of data between the soldiers using the network is maintained."
