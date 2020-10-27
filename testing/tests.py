import unittest
import os
import pandas as pd
from rackio_AI import RackioAI
from rackio import Rackio

class LoadDataTestCase(unittest.TestCase):

    def setUp(self):
        """

        """
        self.app = Rackio()
        self.tpl_filename = os.path.join('..', 'rackio_ai', 'data', 'Leak', 'Leak112.tpl')
        self.pkl_filename = os.path.join('..', 'rackio_ai', 'data', 'pkl_files', 'test_data.pkl')
        self.tpl_filename_not_found = os.path.join('data', 'Leak', 'Leak112.tpl')

    def testLoadTPL(self):
        """

        """
        RackioAI(self.app)
        self.assertTrue(isinstance(RackioAI.load(self.tpl_filename), pd.DataFrame)==True)

    def testLoadDataFrameInPickle(self):

        RackioAI(self.app)
        self.assertTrue(isinstance(RackioAI.load(self.pkl_filename), pd.DataFrame)==True)

    def testLoadFileNotFound(self):

        RackioAI(self.app)
        self.assertRaises(FileNotFoundError, RackioAI.load(self.tpl_filename_not_found), {})

    def testLoadInvalidPickle(self):

        pass

if __name__=='__main__':
    unittest.main()