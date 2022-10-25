import unittest
import numpy as np
import os
from . import RackioAI, get_directory
from rackio_AI import RackioAI, RackioEDA

class DataWrangling(RackioEDA):

    app = RackioAI

    def __init__(self, **kwargs):
        super(DataWrangling, self).__init__(**kwargs)

    def run(self, directory):
        self.directory = directory

        # Functions arguments of the Pipeline
        # This is a list for the arguments of each method of Pipeline/Wrangling class
        func_args = [
            {
                "args": [],
                "kwargs":{'join_files': False}
            },
            {
                "args": [],
                "kwargs":{
                    "KAPPA_POSITION_PIPELINE@19M": "KAPPA_POSITION_POS19M",
                    "KAPPA_POSITION_PIPELINE@58M": "KAPPA_POSITION_POS58M",
                    "PT_POSITION_PIPELINE@19M": "PT_POSITION_POS19M",
                    "PT_POSITION_PIPELINE@58M": "PT_POSITION_POS58M",
                    "PT_POSITION_PIPELINE@1378M": "PT_POSITION_POS1378M"
                }
            },
            {
                "args": [
                    ("KAPPA_POSITION_POS19M", "Compressibility_of_fluid", "1/Pa"),
                    ("KAPPA_POSITION_POS58M", "Compressibility_of_fluid", "1/Pa")
                ],
                "kwargs":{}
            },
            {
                "args": [
                    ("TIME_SERIES", "", "S"),
                    ("PT_POSITION_POS19M", "Pressure", "PA"),
                    ("PT_POSITION_POS58M", "Pressure", "PA"),
                    ("PT_POSITION_POS1378M", "Pressure", "PA")
                ],
                "kwargs":{}
            },
            {
                "args": [],
                "kwargs":{}
            },
            {
                "args": ["TIME_SERIES", "TIMESTAMP"],
                "kwargs": {}
            },
            {
                "args": [],
                "kwargs":{}
            },
            # {
            #     "args": [0.5, "TIMESTAMP"],
            #     "kwargs": {}
            # },
            {
                "args": [],
                "kwargs":{}
            }
        ]

        # Pipeline definition
        self(
            func_args, #Method's arguments
            RackioAI.load, #Methods
            self.rename_columns,
            self.remove_columns,
            self.keep_columns,
            self.add_column,
            self.set_datetime_index,
            self.reset_index,
            # self.resample,
            self.print_report
        )
        self.start(self.directory)

    def add_column(self, data:list):
        r"""
        Documentation here
        """
        new_result = list()

        # print("Se hace la modificación y retornamos el mismo tipo de dato")
        for elem in data:

            df = elem['tpl']
            new_data = np.ones([len(df.values), 1])
            _result = self.insert_columns(df, new_data, column_names=[("FF_01", "Friction Factor", "Adim.")])

            new_result.append(
                {
                    'tpl': _result,
                    'genkey': elem['genkey'],
                    'settings': elem['settings']
                }
            )

        result = new_result

        return result

    def save(self, filename):
        
        self.app.save(self.app.data, filename)


class TestLoadTPLList(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        pass

    def test_load_list_tpl_with_genkey_gasoline_directory(self):
        """

        :return:
        """
        directory = os.path.join(get_directory(os.path.join('Gasoline')))
        # data = RackioAI.load(directory, join_files=False)
        wrangling = DataWrangling()
        wrangling.run(directory)
        # print(f"ETL Result: {wrangling.etl_data}")
