import sys
from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

prompt_command = "What is a castle?"

# Generate answer with prompt
response = generator(prompt_command, max_length=400)

# Print answer
print(response)

print("="*30)
print(response[0]['generated_text'])

# # TODO: Remove after finalizing PRAHOM package #############################
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# # sys.path.append(module_path + "/prahom_wrapper")
# sys.path.append(module_path)
# ############################################################################

# from prahom_wrapper.relation_model import osj_relation
# from prahom_wrapper.prahom_wrapper import prahom_wrapper
# from prahom_wrapper.policy_model import joint_policy_constraint
# from prahom_wrapper.organizational_model import organizational_model, structural_specifications
# from prahom_wrapper.history_model import action_label, joint_histories, joint_history, observation_label
# import numpy as np
# from custom_envs.movingcompany import moving_company_v0
# from typing import Dict, Union

# from custom_envs.movingcompany import moving_company_v0
# from prahom_wrapper.prahom_wrapper import prahom_wrapper

# """
# env = moving_company_v0.raw_env(render_mode="human", size=10, seed=42)

# roles = organizational_model(structural_specifications(roles=["role_0", "role_1", "role2"], ...)
# jt_histories = joint_histories(env.possible_agents).add_joint_history(jth)

# osj_rel = osj_relation(env.possible_agents).link_os(roles, jt_histories, env.possible_agents)

# roles_agents_pc = joint_policy_constraint([osj_rel.get_joint_histories(roles, env.possible_agents)])

# train_env = moving_company_v0.parallel_env(render_mode="grid", size=10, seed=42)
# eval_env = moving_company_v0.parallel_env(render_mode="rgb_array", size=10, seed=42)
# env = prahom_wrapper(env, osj_rel, roles_agents_pc, label_to_obj)

# env.train_under_constraints(train_env=train_env, test_env=eval_env, total_step=0)
# """

# env = moving_company_v0.raw_env(render_mode="human", size=10, seed=42)


# label_to_obj: Dict[Union[observation_label, action_label], object] = {
#     "a0": 0,
#     "a1": 1,
#     "a2": 2,
#     "a3": 3,
#     "a4": 4,
#     "a5": 5,
#     "a6": 6,

#     "o01": np.array([0, 1, 0, 0, 2, 0, 0, 1, 0]),  # 0 -> 1
#     "o02": np.array([0, 5, 0, 0, 2, 0, 0, 1, 0]),  # 1 -> 5
#     "o03": np.array([0, 4, 0, 0, 3, 0, 0, 1, 0]),  # 2 -> 2
#     "o04": np.array([0, 1, 0, 0, 3, 0, 0, 1, 0]),  # 2 -> 2
#     "o05": np.array([0, 1, 0, 0, 3, 0, 0, 4, 1]),  # 3 -> 6
#     "o06": np.array([0, 1, 0, 0, 3, 0, 0, 4, 2]),  # 3 -> 6
#     "o07": np.array([0, 1, 0, 0, 2, 0, 0, 5, 1]),  # -> 0
#     "o08": np.array([0, 1, 0, 0, 2, 0, 0, 5, 2]),  # -> 0
#     "o09": np.array([0, 1, 0, 0, 2, 0, 0, 4, 3]),  # -> 0
#     "o010": np.array([0, 1, 0, 0, 2, 0, 0, 4, 1]),  # -> 0

#     "o11": np.array([1, 0, 0, 5, 2, 1, 0, 0, 0]),  # 1 -> 5
#     "o12": np.array([2, 0, 0, 5, 2, 1, 0, 0, 0]),  # 1 -> 5
#     "o13": np.array([1, 0, 0, 4, 3, 1, 0, 0, 0]),  # 2 -> 4
#     "o14": np.array([2, 0, 0, 4, 3, 1, 0, 0, 0]),  # 2 -> 4
#     "o15": np.array([0, 0, 0, 1, 3, 1, 0, 0, 0]),  # 2 -> 4
#     "o16": np.array([0, 0, 0, 1, 2, 1, 0, 0, 0]),  # 0 -> 3
#     "o17": np.array([0, 0, 1, 1, 3, 4, 0, 0, 0]),  # 3 -> 6
#     "o18": np.array([0, 0, 2, 1, 3, 4, 0, 0, 0]),  # 3 -> 6
#     "o19": np.array([0, 0, 1, 1, 2, 5, 0, 0, 0]),  # -> 0
#     "o110": np.array([0, 0, 2, 1, 2, 5, 0, 0, 0]),  # -> 0
#     "o111": np.array([1, 0, 0, 4, 2, 1, 0, 0, 0]),  # -> 0
#     "o112": np.array([3, 0, 0, 4, 2, 1, 0, 0, 0]),  # -> 0
#     "o113": np.array([0, 0, 1, 1, 2, 4, 0, 0, 0]),  # -> 0
#     "o114": np.array([0, 0, 3, 1, 2, 4, 0, 0, 0]),  # -> 0

#     "o21": np.array([0, 1, 0, 0, 2, 0, 1, 5, 0]),  # 2 -> 5
#     "o22": np.array([0, 1, 0, 0, 2, 0, 2, 5, 0]),  # 2 -> 5
#     "o23": np.array([0, 1, 0, 0, 3, 0, 1, 4, 0]),  # 3 -> 1
#     "o25": np.array([0, 4, 0, 0, 2, 0, 0, 1, 0]),  # 1 -> 2
#     "o26": np.array([0, 1, 0, 0, 2, 0, 1, 4, 0]),  # -> 0
#     "o27": np.array([0, 1, 0, 0, 2, 0, 3, 4, 0])  # -> 0
# }

# jth: joint_history = {'agent_0': ['o02', 'a5', 'o03', 'a2', 'o04', 'a2', 'o04', 'a2', 'o04', 'a2', 'o04', 'a2', 'o06', 'a6', 'o08', 'a0', 'o09', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010', 'a0', 'o010'], 'agent_1': ['o111', 'a0', 'o111', 'a0', 'o111', 'a0', 'o111', 'a0', 'o111', 'a0', 'o111', 'a0', 'o112', 'a0', 'o12', 'a5', 'o14', 'a4', 'o15', 'a4', 'o15', 'a4', 'o15', 'a4', 'o15', 'a4', 'o18', 'a6', 'o110', 'a0', 'o114', 'a0', 'o113', 'a0', 'o113', 'a0', 'o113', 'a0', 'o113', 'a0', 'o113', 'a0', 'o113'], 'agent_2': ['o25', 'a0', 'o25', 'a2', 'o01', 'a2', 'o01', 'a2', 'o01', 'a2', 'o01', 'a2', 'o26', 'a0', 'o26', 'a0', 'o26', 'a0', 'o26', 'a0', 'o26', 'a0', 'o26', 'a0', 'o26', 'a0', 'o27', 'a0', 'o22', 'a5', 'a1', 'o04', 'a1', 'o04', 'a1', 'o04', 'a1', 'o04', 'a1', 'o03', 'a6', 'o02']}

# roles = organizational_model(
#     structural_specifications=structural_specifications(roles=["role_0", "role_1", "role2"], role_inheritance_relations=None, root_groups=None), functional_specifications=None, deontic_specifications=None)
# jt_histories = joint_histories(env.possible_agents).add_joint_history(jth)

# osj_rel = osj_relation(env.possible_agents).link_os(
#     roles, jt_histories, env.possible_agents)

# roles_agents_pc = joint_policy_constraint(
#     [osj_rel.get_joint_histories(roles, env.possible_agents)])

# train_env = moving_company_v0.parallel_env(
#     render_mode="grid", size=10, seed=42)
# eval_env = moving_company_v0.parallel_env(
#     render_mode="rgb_array", size=10, seed=42)
# env = prahom_wrapper(env, osj_rel, roles_agents_pc, label_to_obj)

# env.train_under_constraints(
#     train_env=train_env, test_env=eval_env, total_step=0)

# env.test_trained_model()
