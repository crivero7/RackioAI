import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rackio_AI import Preprocessing, RackioAI
from rackio import Rackio


app = Rackio()
RackioAI(app)

"Filename definition where is the data"
os.chdir('..')
cwd = os.getcwd()
filename = os.path.join(cwd,'rackio_AI','data','pkl_files', 'test_data.pkl')

"Load Data to RackioAI"
RackioAI.load(filename)

"Definition of instrument parameters"
error = [0.0025, 0.0025, 0.0025, 0.0025]
repeteability = [0.001, 0.001, 0.001, 0.001]
lower_limit = [0, 0, 400000, 100000]
upper_limit = [500, 500, 1200000, 600000]
dead_band = [0.001, 0.001, 0.001, 0.001]

"Set Options"
RackioAI.synthetic_data.set_options(error=error, repeteability=repeteability, lower_limit=lower_limit, upper_limit=upper_limit, dead_band=dead_band)

"run synthetic data"
data = RackioAI.synthetic_data(add_WN=True)

"Preprocess class instantiation"
preprocess = Preprocessing(name= 'Kalman Filter', description='test for filter data', problem_type='regression')

"Kalman filter Definition"
kf = preprocess.kalman_filter

"Filter Parameter  alpha=1.0, beta=0.0 No Filter"
kf.alpha = 0.001
kf.beta = 0.2

"Get variable to filter"
FI_01 = data[:,0]
FI_01 = FI_01.reshape(-1,1)

"Optional - set filter init_value"
kf.set_init_value(FI_01[0])

"Applying Kalman filter"
f_FI_01 = np.array([kf(value) for value in FI_01])

"Making DataFrame to plot data and data filtered"
f_FI_01 = f_FI_01.reshape((-1,1))
new_df = pd.DataFrame(np.concatenate([f_FI_01, FI_01], axis=1), columns=['f_FI_01', 'FI_01'])

"Plotting result"
new_df.plot(kind='line', y=['f_FI_01', 'FI_01'])
plt.show(block=True)