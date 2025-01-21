# Classification of multi-agent reinforcement learning (MARL) algorithms along with common parameters:

### 1. Independent Learning Algorithms

In these algorithms, each agent learns independently by treating other agents as part of the environment. They are often extensions of single-agent RL algorithms.

#### Examples

* Independent Q-Learning (IQL)
* Independent DQN (IDQN)

#### Common Parameters

* **Learning rate (α)**: Determines how much new information overrides old information.
* **Discount factor (γ)**: Weighs the importance of future rewards versus immediate rewards.
* **Epsilon (ε)**: Parameter for exploration-exploitation policy (ε-greedy).

### 2. Cooperative Learning Algorithms

These algorithms are designed for environments where agents need to cooperate to achieve a common goal. They often use shared rewards and common policies.

#### Examples

* Cooperative Q-Learning
* Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

#### Common Parameters

* **Learning rate**: For updating value functions and policies.
* **Discount factor**: For calculating cumulative rewards.
* **Epsilon (ε)**: Parameter for exploration-exploitation policy.
* **Replay buffer size**: Size of the buffer used to store experiences.
* **Batch size**: Size of the mini-batches used for learning.
* **Communication cost (if applicable)**: Cost associated with communication between agents.

### 3. Competitive Learning Algorithms

These algorithms are used in environments where agents compete against each other. They must account for the actions of adversarial agents.

#### Examples

* Minimax Q-Learning
* Multi-Agent Deep Q-Network (MADQN)

#### Common Parameters

* **Learning rate**: For updating value functions.
* **Discount factor**: For calculating cumulative rewards.
* **Epsilon (ε)**: Parameter for exploration-exploitation policy.
* **Batch size**: Size of the mini-batches used for learning.
* **Replay buffer size**: Size of the buffer used to store experiences.

### 4. Policy-Based Methods

These algorithms learn policies directly for each agent, often using approaches such as policy gradient learning.

#### Examples

* Multi-Agent Proximal Policy Optimization (MAPPO)
* Multi-Agent Trust Region Policy Optimization (MATRPO)

#### Common Parameters

* **Policy learning rate**: For updating the policy.
* **Value learning rate**: For updating the value function.
* **Discount factor**: For calculating cumulative rewards.
* **Entropy coefficient**: To encourage exploration by adding an entropy term to the loss function.
* **Clip range (for PPO)**: Clip range for probability ratios in PPO.

### 5. Multi-Agent Actor-Critic Methods

These algorithms combine value-based and policy-based approaches using actors and critics for each agent or a shared critic.

#### Examples

* Counterfactual Multi-Agent (COMA)
* Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
* Centralized Critic with Decentralized Actors (CCDA)

#### Common Parameters

* **Actor learning rate**: For updating the policy.
* **Critic learning rate**: For updating the value function.
* **Discount factor**: For calculating cumulative rewards.
* **Entropy coefficient**: To encourage exploration.
* **Soft update parameter (τ)**: For soft updates of target networks.
* **Communication cost**: Cost associated with communication between agents (if applicable).

### 6. Value Decomposition Methods

These algorithms decompose the global team value into individual agent values to facilitate coordinated learning.

#### Examples

* Value-Decomposition Networks (VDN)
* QMIX

#### Common Parameters

* **Learning rate**: For updating value functions and policies.
* **Discount factor**: For calculating cumulative rewards.
* **Replay buffer size**: Size of the buffer used to store experiences.
* **Batch size**: Size of the mini-batches used for learning.
* **Entropy coefficient**: To encourage exploration.

### 7. Multi-Agent Planning Algorithms

These algorithms use models of the environment and agent interactions to plan coordinated action sequences.

#### Examples

* Multi-Agent Model-Based Reinforcement Learning
* Dyna-style multi-agent planning

#### Common Parameters

* **Learning rate**: For updating value functions and policies.
* **Discount factor**: For calculating cumulative rewards.
* **Replay buffer size**: Size of the buffer used to store experiences.
* **Number of planning updates**: Number of policy updates per planning cycle.
* **Model parameters**: Parameters for the environment models used.

This classification covers the main approaches in MARL and their typical parameters, providing an overview of the methods and important considerations for each type of algorithm.
