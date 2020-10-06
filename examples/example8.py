import os
import pandas as pd
import numpy as np
from rackio_AI import RackioAI
from rackio import Rackio


app = Rackio()

RackioAI(app)


"Data load"
os.chdir('../..')
filename = os.path.join(os.getcwd(), 'iDetectFugas','data', 'DataToTest.pkl')

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
data = RackioAI.synthetic_data(decalibrations=0, sensor_drift=0, excesive_noise=0, frozen_data=10, outliers=0, out_of_range=10, add_WN=True)

data = pd.DataFrame(data, columns=variable_names)

print('done')