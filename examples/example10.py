from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

df = RackioAI.load_test_data('Leak')
df.info()

df2 = RackioAI.load_test_data('Leak', 'Leak111.tpl')
df2.info()