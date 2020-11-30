import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

filename = os.path.join('..', 'data')

data = RackioAI.load(filename)

print(data)
