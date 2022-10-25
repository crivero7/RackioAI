import unittest
import os
from . import get_directory


class TestOSPath(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        pass

    def test_os_path(self):
        r"""
        Documentation here
        """
        directories = ["Gasoline", "Leak", "Steady State", "D0", "TPL", "01"]
        filename = os.path.join(get_directory(), *directories)
        genkey_filename = filename.split(os.path.sep)
        genkey_filename.pop(-2)
        genkey_filename = os.path.join(os.path.sep, *genkey_filename) + '.genkey'
        print(f"Genkey Filename: {genkey_filename}")
        