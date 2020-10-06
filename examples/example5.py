import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

os.chdir('..')
cwd = os.getcwd()
filename = os.path.join(cwd,'data', 'Stroke de fuga 10s','Case112.tpl')

RackioAI.load_data(filename)

df = RackioAI.convert_data_to('dataframe')
