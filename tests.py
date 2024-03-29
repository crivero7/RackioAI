from unittest import TestLoader, TestSuite, TextTestRunner
from rackio_AI.tests.tests_load import TestLoadData
from rackio_AI.tests.tests_genkey import TestGenkey
from rackio_AI.tests.tests_load_tpl_with_genkey import TestLoad
from rackio_AI.tests.tests_load_pkl import TestLoadPKL
from rackio_AI.tests.tests_os_path import TestOSPath
from rackio_AI.tests.tests_etl_pipeline import TestLoadTPLList


def suite():
    r"""
    Documentation here
    """
    tests = list()
    suite = TestSuite()
    tests.append(TestLoader().loadTestsFromTestCase(TestLoadData))
    tests.append(TestLoader().loadTestsFromTestCase(TestGenkey))
    tests.append(TestLoader().loadTestsFromTestCase(TestLoad))
    tests.append(TestLoader().loadTestsFromTestCase(TestLoadTPLList))
    tests.append(TestLoader().loadTestsFromTestCase(TestLoadPKL))
    tests.append(TestLoader().loadTestsFromTestCase(TestOSPath))
    suite = TestSuite(tests)
    return suite


if __name__ == '__main__':

    runner = TextTestRunner()
    runner.run(suite())
