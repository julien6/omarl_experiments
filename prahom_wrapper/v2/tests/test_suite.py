import unittest
from test_history_model import test_history_subset

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_history_subset))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
