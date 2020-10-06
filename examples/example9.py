import os
import pandas as pd
import numpy as np
from rackio_AI.preprocess import Preprocess
from rackio_AI import RackioAI
from rackio import Rackio
import matplotlib.pyplot as plt


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
data = RackioAI.synthetic_data(decalibrations=0, sensor_drift=0, excesive_noise=0, frozen_data=0, outliers=0, out_of_range=0, add_WN=True)

data = pd.DataFrame(data, columns=variable_names)

"Applying Kalman Filter"
preprocess = Preprocess(name= 'Kalman Filter', description='test for filter data', problem_type='regression')
RackioAI.append_preprocess_model(preprocess)

kf = preprocess.preprocess.kalman_filter

FI_01 = data.iloc[:,0].values
FI_01 = FI_01.reshape(-1,1)
f_FI_01 = np.zeros(len(FI_01))

for count, value in enumerate(FI_01):
    f_FI_01[count] = kf(value)

f_FI_01 = f_FI_01.reshape((-1,1))
new_df = pd.DataFrame(np.concatenate([f_FI_01, FI_01], axis=1), columns=['f_FI_01', 'FI_01'])
new_df.plot(kind='line', y=['f_FI_01', 'FI_01'])

plt.show(block=True)

