# Organization oriented Multi-Agent Reinforcement Learning

Organization oriented Multi-Agent Reinforcement Learning (OMARL) is a broad topic gathering research that specifically aims at integrating organization model of a Multi-Agent System (MAS) into Multi-Agent Reinforcement Learning (MARL) to improve its explainability and handability for MAS.

In a regular MARL scenario, agents' policies (their rules) are updated via an algorithm until reaching a goal. That process results in a set of trained agents who collaborates to fullfil a goal.

Unfortunately, when getting interested to have an organization point-of-view, the "trained" or "on training" joint-policy gives "learned" rules that could be mixing the individual, social (interactions in neighborhood) or even collective level (interactions with majority).

Even though some works have pushing forward interest for emergent collective phenomena in MARL, none has considered using an organization model to characterize them (as cooperation schemes, roles, missions...); or even driving the agents' learning through extra organizational specifications.

OMARL's original interest is to have an explicit way to view and act on a set learning agents not only at individual but also social and collective levels.

## Dec-POMDP $\mathcal{M}OISE^+$ OMARL (DMO)

*Dec-POMDP $\mathcal{M}OISE^+$ OMARL* (DMO) is first attempt to implement a process that fits OMARL goal as for linking:
 - **Decentralized Partially Observable Markov Decision Process** (Dec-PODMP): a model for using MARL in any formalized scenario; 
 - **$\mathcal{M}OISE^+$**: an organizational model that relies on structural, functional and deontic specifications to describe a MAS organization

We proposed the *Partially Action-based $\mathcal{M}OISE^+$ Identification DMO* (PAMID), a DMO process that allows:
 - Getting the resulting organizational specifications out of a given joint-policy by linking actions with known organizational specification; 
 - Constraining the agents' training to respect some organizational specifications by restricting available agents' actions at each step

## PAMID Gym-wrapper

The PAMID process resulted in an algorithm we integrated within the Pettingzoo interface under the form of a Gym wrapper for easy use. That Gym wrapper is to be used to apply ORMARL on environment that requires collaboration and organization among agents.

A typical workflow would consist in:

1) Associating actions with known organizational specifications
2) Defining specifications to respect as extra constraints during training
3) Launching the training over several iterations (or epoch) to get several successful joint-policies
4) Extracting some organizational specifications out of thee joint-policies concerning the emergent organization

## Installation

1) Clone current repo in a safe place
   
2) Create a new clean virtual python environment:

```cd ~; mkdir python-envs; cd python-envs; python -m venv pamid; source pamid/bin/activate; cd ~```

1) In cloned repo, type:

```python install requirements.txt```

4) Then, type:

```pip install -e .```

## Tutorial: Predator-prey with communication

Source link: https://pettingzoo.farama.org/environments/mpe/simple_world_comm/

![alt text](https://github.com/julien6/omarl_experiments/blob/main/images/mpe_simple_world_comm.gif?raw=true)

Simple World Comm is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By default, there is 1 good agent, 3 adversaries and 2 obstacles.

Additionally there is food (small blue balls) that the good agents are rewarded for being near, there are ‘forests’ that hide agents inside from being seen, and there is a ‘leader adversary’ that can see the agents at all times and can communicate with the other adversaries to help coordinate the chase. By default, there are 2 good agents, 3 adversaries, 1 obstacles, 2 foods, and 2 forests.

In particular, the good agents reward, is -5 for every collision with an adversary, -2 x bound by the bound function described in simple_tag, +2 for every collision with a food, and -0.05 x minimum distance to any food. The adversarial agents are rewarded +5 for collisions and -0.1 x minimum distance to a good agent.

### PAMID Gym-wrapper use:

Basic example

```python
from pettingzoo.mpe import simple_world_comm_v3
from omarl_experiments import PamidWrapper

env = simple_world_comm_v3.parallel_env(render_mode="human")

action_to_specs = ...
training_specs = ...

# wrapping the initial environment to use PAMID
env = PamidWrapper(env, action_to_specs, training_specs, unknown_specs_inference=True, pca_output=True)

observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: policies[] env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

# getting pamid output
env.pamid_render_pca() # print the joint-policies in a PCA for comparison between agents (help to analyze similar agents behaviors)
trained_specs, agent_to_specs = env.pamid_specs()

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

TODO: showing the learning curve with and without OMARL, time to converge, stability...

Finally, we can extract the organizational specifications out of the resulting

```python
env.pamid_render_pca() # print the joint-policies in a PCA for comparison
trained_specs, agent_to_specs = env.pamid_specs()
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

## Tutorial: Knights Archers Zombies

![alt text](https://github.com/julien6/omarl_experiments/blob/main/images/butterfly_knights_archers_zombies.gif?raw=true)

TODO...
