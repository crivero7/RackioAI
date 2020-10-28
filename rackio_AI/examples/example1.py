from rackio import Rackio
from rackio_AI import RackioAI

app = Rackio()


RackioAI(app)

"Rackio"
print(RackioAI.app.__class__.__name__)
print('===================')
"RackioAI"
print(RackioAI.__class__.__name__)