import unittest
from prahom_wrapper.v2.prahom_wrapper.history_model import history_subset, joint_history_subset, osh_relations, os_factory, history_factory, joint_history_factory


class test_prahom_wrapper(unittest.TestCase):

    def setUp(self):
        pz_env = raw_env()
        pz_env = prahom_wrapper(pz_env)

        pz_env.train_under_constraints(env_creator=env_creator,
                                    osh_model_constraint=organizational_model(
                                        structural_specifications=structural_specifications(
                                            roles={"role_0": history_subset()},
                                            role_inheritance_relations=None,
                                            root_groups=None),
                                        functional_specifications=functional_specifications(
                                            social_scheme=social_scheme(
                                                goals={
                                                    "goal_0": history_subset()},
                                                missions=None,
                                                goals_structure=None,
                                                mission_to_goals=None,
                                                mission_to_agent_cardinality=None),
                                            social_preferences=None),
                                        deontic_specifications=None),
                                    constraint_integration_mode=constraints_integration_mode.CORRECT,
                                    algorithm_configuration={}
                                    )

        om = pz_env.generate_organizational_specifications(
            use_kosia=False, use_gosia=True, gosia_configuration=gosia_configuration(generate_figures=True))
        print(om)

    def test_add_observation_action(self):
        # self.assertEqual(self.calc.addition(2, 3), 5)
        pass

    def test_add_action_observation(self):
        pass

    def test_add_history(self):
        pass

    def tearDown(self):
        del self.hs


if __name__ == '__main__':
    unittest.main()
