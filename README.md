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

## Tutorial: Knights Archers Zombies

![alt text](https://github.com/julien6/omarl_experiments/images/butterfly_knights_archers_zombies.gif?raw=true)

