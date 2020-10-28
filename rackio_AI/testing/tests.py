import unittest
import os
import pandas as pd
from rackio_AI import RackioAI
from rackio_AI.data import get_directory
from rackio import Rackio

class LoadDataTestCase(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        self.app = Rackio()
        self.tpl_filename = os.path.join(get_directory('Leak'), 'Leak112.tpl')
        self.pkl_filename = os.path.join(get_directory('pkl_files'), 'test_data.pkl')
        self.tpl_filename_not_found = os.path.join(get_directory('Leak'), 'Leak212.tpl')
        self.pkl_filename_not_found = os.path.join(get_directory('pkl_files'), 'Leak212.tpl')
        self.no_valid_pkl_file = os.path.join('../..', 'rackio_ai', 'data', 'pkl_files', 'no_valid_RackioAI_file.pkl')

    def testLoadTPL(self):
        """

        :return:
        """
        RackioAI(self.app)
        self.assertTrue(isinstance(RackioAI.load(self.tpl_filename), pd.DataFrame)==True)

    def testLoadDataFrameInPickle(self):
        """

        :return:
        """
        RackioAI(self.app)
        self.assertTrue(isinstance(RackioAI.load(self.pkl_filename), pd.DataFrame)==True)

    def testLoadFileNotFound(self):
        """

        :return:
        """
        RackioAI(self.app)
        with self.assertRaises(FileNotFoundError):
            RackioAI.load(self.tpl_filename_not_found)

        with self.assertRaises(FileNotFoundError):
            RackioAI.load(self.pkl_filename_not_found)

    def testLoadNoValidPickleToRackioAI(self):
        """

        :return:
        """
        RackioAI(self.app)
        with self.assertRaises(ModuleNotFoundError):
            RackioAI.load(self.no_valid_pkl_file)

if __name__=='__main__':
    unittest.main()