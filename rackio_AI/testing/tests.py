import unittest
import os
import pandas as pd
import numpy as np
from rackio_AI import RackioAI, Preprocessing, get_directory


class LoadDataTestCase(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        pass

    def testLoadTPL(self):
        """

        :return:
        """
        filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        df = RackioAI.load(filename)
        print(df.head(10))

#     def testLoadDataFrameInPickle(self):
#         """

#         :return:
#         """
#         RackioAI(self.app)
#         self.assertTrue(isinstance(RackioAI.load(self.pkl_filename), pd.DataFrame) == True)

#     def testLoadFileNotFound(self):
#         """

#         :return:
#         """
#         RackioAI(self.app)
#         with self.assertRaises(FileNotFoundError):
#             RackioAI.load(self.tpl_filename_not_found)

#         with self.assertRaises(FileNotFoundError):
#             RackioAI.load(self.pkl_filename_not_found)

#     def testLoadNoValidPickleToRackioAI(self):
#         """

#         :return:
#         """
#         RackioAI(self.app)
#         with self.assertRaises(ModuleNotFoundError):
#             RackioAI.load(self.no_valid_pkl_file)


# class SplitterTestCase(unittest.TestCase):
#     def setUp(self):
#         """

#         :return:
#         """
#         self.app = Rackio()

#     def testRandomProportion(self):
#         """

#         :return:
#         """
#         RackioAI(self.app)
#         preprocess = Preprocessing(name='Preprocess model name', description='preprocess for data',
#                                    problem_type='regression')
#         X, y = np.arange(20000).reshape((10000, 2)), range(10000)
#         # Number of test to do
#         number_of_tests = 100
#         for i in range(number_of_tests):
#             # Generate amount of parameter to pass as function argument, if rand_int=1, you pass only train_size parameter
#             # rand_int=2, you pass train_size and test_size parameters
#             # rand_int=3, you pass train_size, test_size and validation size parameters
#             rand_int = np.random.randint(3) + 1
#             # Generate proportion split for train_test_validation parameters
#             array = np.random.random(rand_int)
#             # Parameter definition
#             keys = ['train_size', 'test_size', 'validation_size']
#             # Create dictionary of normalized parameter
#             train_test_validation_size = {'{}'.format(keys[count]): value / np.sum(array) if rand_int > 1 else value for
#                                           count, value in enumerate(array)}
#             # Do the test
#             self.assertIsInstance(preprocess.splitter.split(X, y, random_state=0, **train_test_validation_size), list)


if __name__ == '__main__':
    # unittest.main()
    filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
    df = RackioAI.load(filename)
    print(df.head(10))
