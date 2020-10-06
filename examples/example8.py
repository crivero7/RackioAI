import os
import pandas as pd
import numpy as np
from rackio_AI import RackioAI
from rackio import Rackio


app = Rackio()

RackioAI(app)


"Data load"
os.chdir('..')
cwd = os.getcwd()

base_path = os.path.join(cwd,'rackio_AI','data','pkl_files')

filename = os.path.join(base_path, 'test_data.pkl')

RackioAI.data = filename

variable_names = RackioAI.data.columns.to_list()

"Definition of instrument parameters"
error = np.array([0.0025,
                  0.0025,
                  0.0025,
                  0.0025])

repeteability = np.array([0.001,
                          0.001,
                          0.001,
                          0.001])

lower_limit = np.array([0,
                       0,
                       400000,
                       100000])

upper_limit = np.array([500,
                       500,
                       1200000,
                       600000])

dead_band = np.array([0.001,
                     0.001,
                     0.001,
                     0.001])

"Set Options"
RackioAI.synthetic_data.set_options(error=error, repeteability=repeteability, lower_limit=lower_limit, upper_limit=upper_limit, dead_band=dead_band)

"run synthetic data"
data = RackioAI.synthetic_data(decalibrations=0, sensor_drift=0, excesive_noise=0, frozen_data=2, outliers=0, out_of_range=1, add_WN=True)

data = pd.DataFrame(data, columns=variable_names)

print('done')