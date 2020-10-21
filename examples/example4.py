import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

os.chdir('..')
cwd = os.getcwd()

base_path = os.path.join(cwd,'rackio_AI','data')

filename = os.path.join(base_path, 'Leak')

RackioAI.data = filename

url_to_save = os.path.join(base_path,'name.csv')
RackioAI.convert_data_to('csv', filename=url_to_save)
