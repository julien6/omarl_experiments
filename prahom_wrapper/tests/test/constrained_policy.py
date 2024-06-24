import numpy as np
from sklearn.base import BaseEstimator
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.agents.ppo import PPOTorchPolicy
from sklearn.tree import DecisionTreeClassifier


class CompositePolicy(Policy):
    def __init__(self, observation_space, action_space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)
        self.rllib_policy = PPOTorchPolicy(
            observation_space, action_space, config)
        self.decision_tree = config["decision_tree"]

    def compute_actions(
            self,
            obs_batch: TensorType,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs):

        # Convert observations to a numpy array for the decision tree
        obs_batch_np = obs_batch.numpy() if hasattr(
            obs_batch, 'numpy') else np.array(obs_batch)

        # Predict actions using the decision tree
        decision_tree_actions = self.decision_tree.predict(obs_batch_np)

        # Determine which actions come from the decision tree
        tree_action_mask = ~np.isnan(decision_tree_actions)

        # Compute actions using the RLlib policy for observations not handled by the decision tree
        rllib_obs_batch = obs_batch[~tree_action_mask]
        rllib_actions, rllib_state_out, rllib_extra_fetches = self.rllib_policy.compute_actions(
            rllib_obs_batch, state_batches, prev_action_batch, prev_reward_batch, info_batch, episodes, **kwargs)

        # Combine actions from the decision tree and the RLlib policy
        actions = np.full_like(decision_tree_actions,
                               fill_value=np.nan, dtype=np.float32)
        actions[tree_action_mask] = decision_tree_actions[tree_action_mask]
        actions[~tree_action_mask] = rllib_actions

        return actions, state_batches, rllib_extra_fetches

    def learn_on_batch(self, samples: SampleBatch):
        return self.rllib_policy.learn_on_batch(samples)

    def get_weights(self):
        return self.rllib_policy.get_weights()

    def set_weights(self, weights):
        self.rllib_policy.set_weights(weights)
