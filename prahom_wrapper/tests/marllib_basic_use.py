# import gym.spaces.utils
# from custom_envs.ce import RLlibCE
# from custom_envs.ce_fcoop import RLlibCE_FCOOP
# from marllib.envs.base_env import ENV_REGISTRY
# from marllib.envs.global_reward_env import COOP_ENV_REGISTRY


from marllib import marl

# prepare the environment academy_pass_and_shoot_with_keeper
# env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
env = marl.make_env(environment_name="mpe",
                    map_name="simple_spread", force_coop=True)

# can add extra env params. remember to check env configuration before use
# env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# can add extra algorithm params. remember to check algo_config hyperparams before use
# mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=1,
          num_workers=10, share_policy='all', checkpoint_freq=500)

# rendering
mappo.render(
    env, model,
    local_mode=True,
    restore_path={'params_path': "checkpoint/params.json",
                  'model_path': "checkpoint/checkpoint-10"}
)


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
