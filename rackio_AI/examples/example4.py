import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

base_path = os.path.join('..', 'data')

filename = os.path.join(base_path, 'Leak')

RackioAI.load(filename)

url_to_save = os.path.join(base_path, 'name.csv')
RackioAI.reader.tpl.to('csv', filename=url_to_save)
