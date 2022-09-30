from unittest import TestLoader, TestSuite, TextTestRunner
from rackio_AI.tests.tests_load import TestLoadData
from rackio_AI.tests.tests_genkey import TestGenkey


def suite():
    r"""
    Documentation here
    """
    tests = list()
    suite = TestSuite()
    tests.append(TestLoader().loadTestsFromTestCase(TestLoadData))
    tests.append(TestLoader().loadTestsFromTestCase(TestGenkey))
    suite = TestSuite(tests)
    return suite


if __name__=='__main__':
    
    runner = TextTestRunner()
    runner.run(suite())
