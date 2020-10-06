import os
from rackio import Rackio
from rackio_AI import RackioAI
from rackio_AI.preprocess import Preprocess

app = Rackio()


RackioAI(app)

os.chdir('../../iDetectFugas')
cwd = os.getcwd()
filename = os.path.join(cwd,'data','olgaData','Bloque con fugas', 'Stroke de fuga 10s','Case010.tpl')

RackioAI.set_data(filename)
data = RackioAI.convert_data_to('dataframe')

preprocessing = Preprocess(name='Model 1', description='data preprocessing', problem_type='regression')
RackioAI.append_preprocess_model(preprocessing)

pre = RackioAI.get_preprocess('Model 1')

pre('scaler', data)