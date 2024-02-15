from pettingzoo.mpe import simple_world_comm_v3

env = simple_world_comm_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

print(env.action_space("adversary_0"))

# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#     observations, rewards, terminations, truncations, infos = env.step(actions)
# env.close()
