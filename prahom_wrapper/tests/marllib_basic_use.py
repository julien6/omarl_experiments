# import gym.spaces.utils
# from custom_envs.ce import RLlibCE
# from custom_envs.ce_fcoop import RLlibCE_FCOOP
# from marllib.envs.base_env import ENV_REGISTRY
# from marllib.envs.global_reward_env import COOP_ENV_REGISTRY


from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='mpe')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 1000000}, share_policy='group')

# ===========================

# # register new env
# ENV_REGISTRY["custom_envs"] = RLlibCE
# COOP_ENV_REGISTRY["custom_envs"] = RLlibCE_FCOOP

# # initialize env
# env = marl.make_env(environment_name="custom_envs", map_name="moving_company", force_coop=False, abs_path="../../../custom_envs/ce.yaml",
#                     size=10, seed=42, render_mode="rgb_array")

# # pick mappo algorithms
# mappo = marl.algos.mappo(hyperparam_source="test")

# # build agent model based on env + algorithms + user preference
# model = marl.build_model(
#     env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

# # start learning
# mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
#           num_workers=2, share_policy='all', checkpoint_freq=50)
