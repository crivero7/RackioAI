import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

df = RackioAI.test_data(name='Leak')
df.info()