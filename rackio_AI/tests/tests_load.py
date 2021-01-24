import unittest
import os
import pandas as pd
from rackio_AI import get_directory, RackioAI


class LoadDataTestCase(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        pass

    def test_load_tpl_file(self):
        """

        :return:
        """
        filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        df = RackioAI.load(filename)

    def test_load_tpl_directory(self):
        """

        :return:
        """
        directory = os.path.join(get_directory('Leak'))
        df = RackioAI.load(directory)

    def test_load_csv_file(self):
        """

        :return:
        """
        filename = os.path.join(get_directory('csv'), "standard", "username-password-recovery-code.csv")
        RackioAI.load(filename, ext=".csv", delimiter=";", header=0)
    
    def test_load_csv_directory(self):
        """

        :return:
        """
        directory = os.path.join(get_directory('csv'), "standard")
        RackioAI.load(directory, ext=".csv", delimiter=";", header=0)

    def test_load_csv_hysys(self):
        """

        :return:
        """
        directory = os.path.join(get_directory('csv'), "Hysys")
        df = RackioAI.load(directory, ext=".csv", _format="hysys")

    def test_load_csv_vmgsim(self):
        """

        :return:
        """
        directory = os.path.join(get_directory('csv'), "VMGSim")
        df = RackioAI.load(directory, ext=".csv", _format="vmgsim")

    def test_load_pkl_file(self):
        """

        :return:
        """
        filename = os.path.join(get_directory('pkl_files'), 'test_data.pkl')
        df = RackioAI.load(filename)

    def test_load_pkl_directory(self):
        """

        :return:
        """
        filename = os.path.join(get_directory('pkl_files'))
        df = RackioAI.load(filename, ext=".pkl")


if __name__ == '__main__':
    
    unittest.main()
