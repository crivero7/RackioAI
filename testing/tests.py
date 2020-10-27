import unittest
import os
from rackio_AI import RackioAI
from rackio import Rackio

class LoadDataTestCase(unittest.TestCase):

    def setUp(self):
        """

        """
        pass

    def testLoadTPL(self):
        """

        """
        self.app = Rackio()
        filename = os.path.join('..', 'rackio_ai', 'data', 'Leak', 'Leak112.tpl')
        RackioAI(self.app)
        self.assertTrue(RackioAI.load(filename) == 'Hola')

    def testLoadDataFrameInPickle(self):

        pass

    def testLoadInvalidFormatFiles(self):

        pass

    def testLoadInvalidPickle(self):

        pass

if __name__=='__main__':
    unittest.main()