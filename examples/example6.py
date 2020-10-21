import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

os.chdir('..')
cwd = os.getcwd()
# filename is a Directory, from that directory it will load all .tpl files
filename = os.path.join(cwd, 'rackio_AI', 'data', 'Leak')

data = RackioAI.load_data(filename)

print(data)