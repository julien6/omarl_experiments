# import functools
# from absl import app
# from absl import flags
# from dm_control.locomotion import soccer
# from dm_control import viewer
# from tf_agents.policies import random_tf_policy
# import numpy as np


# FLAGS = flags.FLAGS

# flags.DEFINE_enum("walker_type", "BOXHEAD", ["BOXHEAD", "ANT", "HUMANOID"],
#                   "The type of walker to explore with.")
# flags.DEFINE_bool(
#     "enable_field_box", True,
#     "If `True`, enable physical bounding box enclosing the ball"
#     " (but not the players).")
# flags.DEFINE_bool("disable_walker_contacts", False,
#                   "If `True`, disable walker-walker contacts.")
# flags.DEFINE_bool(
#     "terminate_on_goal", False,
#     "If `True`, the episode terminates upon a goal being scored.")


# def main(argv):

#     if len(argv) > 1:
#         raise app.UsageError("Too many command-line arguments.")

#     env = soccer.load(team_size=2,
#                       walker_type=soccer.WalkerType[FLAGS.walker_type],
#                       disable_walker_contacts=FLAGS.disable_walker_contacts,
#                       enable_field_box=FLAGS.enable_field_box,
#                       keep_aspect_ratio=True,
#                       terminate_on_goal=FLAGS.terminate_on_goal)

#     action_spec = env.action_spec()

#     # Define a uniform random policy.
#     def random_policy(time_step):
#       del time_step  # Unused.
#       return np.random.uniform(low=action_spec.minimum,
#                                high=action_spec.maximum,
#                                size=action_spec.shape)

#     # random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
#     #                                                 env.action_spec())

#     viewer.launch(environment_loader=env, policy=random_policy)


# if __name__ == "__main__":
#     app.run(main)

from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="humanoid", task_name="stand")
action_spec = env.action_spec()

# Define a uniform random policy.


def random_policy(time_step):
    del time_step  # Unused.
    return np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)


# Launch the viewer application.
viewer.launch(env, policy=random_policy)
