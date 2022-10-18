import unittest
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
                "args": [
                    ("TIME_SERIES", "", "S"),
                    # ("PT_POSITION_POS19M", "Pressure", "PA"),
                    # ("PT_POSITION_POS58M", "Pressure", "PA"),
                    # ("PT_POSITION_POS1378M", "Pressure", "PA"),
                    # ("TM_POSITION_POS19M", "Fluid_temperature", "C"),
                    # ("TM_POSITION_POS58M", "Fluid_temperature", "C"),
                    # ("TM_POSITION_POS1378M", "Fluid_temperature", "C"),
                    # ('GT_POSITION_POS19M', 'Total_mass_flow', 'KG/S'),
                    # ('GT_POSITION_POS58M', 'Total_mass_flow', 'KG/S'),
                    # ('GT_POSITION_POS1378M', 'Total_mass_flow', 'KG/S'),
                    # ("GTLEAK_LEAK_LEAK", "Leakage_total_mass_flow_rate", "KG/S"),
                    # ("ACMLK_LEAK_LEAK", "Leakage_accumulated_released_mass", "KG"),
                    # ("PTLKUP_LEAK_LEAK",
                    #  "Pressure_at_the_position_where_Leak_is_positioned", "PA"),
                    # ("PUMPSPEED_PUMP_PUMP", "Pump_speed", "RPM"),
                ],
                "kwargs":{}
            },
            # {
            #     "args": [resample, ("TIME_SERIES", "", "S")],
            #     "kwargs": {}
            # },
        ]

        # Pipeline definition
        self(
            func_args, #Method's arguments
            RackioAI.load, #Methods
            self.keep_columns,
            # self.resample,
        )
        self.start(self.directory)

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
