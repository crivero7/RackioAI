import unittest
import os
from . import RackioAI, get_directory
from pandas import DataFrame


class TestLoad(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        pass

    def test_load_tpl_gasoline_directory(self):
        """

        :return:
        """
        directory = os.path.join(get_directory(os.path.join('Gasoline')))
        data = RackioAI.load(directory, join_files=False)

        with self.subTest("Testing List RackioAI.load"):

            self.assertIsInstance(data, list)

        with self.subTest("Testing Number of Items"):

            self.assertEqual(len(data), 19)

        case = data[0]

        with self.subTest("Testing Case as dict"):

            self.assertIsInstance(case, dict)

        with self.subTest("Testing Case structure"):

            self.assertListEqual(list(case.keys()), [
                                 'tpl', 'genkey', 'settings'])

        with self.subTest("Testing TPL instance"):

            self.assertIsInstance(case['tpl'], DataFrame)

        with self.subTest("Testing genkey instance"):

            self.assertIsInstance(case['genkey'], dict)

        with self.subTest("Testing settings instance"):

            self.assertIsInstance(case['settings'], dict)
