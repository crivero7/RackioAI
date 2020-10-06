import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

os.chdir('..')
cwd = os.getcwd()
filename = os.path.join(cwd,'data','Stroke de fuga 10s')

data = RackioAI.load_data(filename)

url_to_save = os.path.join(cwd,'data','Stroke de fuga 10s','name.csv')
RackioAI.convert_data_to('csv', filename=url_to_save)
