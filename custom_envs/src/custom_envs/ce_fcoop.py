from custom_envs.ce import RLlibCE

legal_scenarios = ["moving_company"]


class RLlibCE_FCOOP(RLlibCE):

    def __init__(self, env_config):
        if env_config["map_name"] not in legal_scenarios:
            raise ValueError("must in: 1.moving_company")
        super().__init__(env_config)

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        reward = 0
        for key in r.keys():
            reward += r[key]
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = reward/self.num_agents
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info
