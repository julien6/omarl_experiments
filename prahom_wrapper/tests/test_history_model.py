import unittest
from prahom_wrapper.v2.prahom_wrapper.history_model import history_subset, joint_history_subset, osh_relations, os_factory, history_factory, joint_history_factory


class test_history_subset(unittest.TestCase):

    def setUp(self):
        self.hs = history_subset()

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
