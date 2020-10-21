import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

os.chdir('..')
cwd = os.getcwd()
filename = os.path.join(cwd, 'rackio_AI', 'data', 'Leak', 'Leak112.tpl')

data = RackioAI.load_data(filename)

print(data)