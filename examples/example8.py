import os
import pandas as pd
from rackio_AI import RackioAI
from rackio import Rackio


app = Rackio()

RackioAI(app)


"Data load"
os.chdir('..')
cwd = os.getcwd()
filename = os.path.join(cwd,'rackio_AI','data','pkl_files', 'test_data.pkl')

RackioAI.load(filename)

variable_names = RackioAI.data.columns.to_list()

"Definition of instrument parameters"
error = [0.0025, 0.0025, 0.0025, 0.0025]
repeteability = [0.001, 0.001, 0.001, 0.001]
lower_limit = [0, 0, 400000, 100000]
upper_limit = [500, 500, 1200000, 600000]
dead_band = [0.001, 0.001, 0.001, 0.001]

"Set Options"
RackioAI.synthetic_data.set_options(error=error, repeteability=repeteability, lower_limit=lower_limit, upper_limit=upper_limit, dead_band=dead_band)

"run synthetic data"
data = RackioAI.synthetic_data(frozen_data=2, out_of_range=1, add_WN=True, view=True, columns=[0,1,2,3], duration={'min': 20, 'max': 100})

data = pd.DataFrame(data, columns=variable_names)

print('done')