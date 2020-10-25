from rackio_AI import RackioEDA, RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

EDA1 = RackioEDA(name= 'EDA1', description='Object 1 Exploratory Data Analysis')

EDA2 = RackioEDA(name= 'EDA2', description='Object 2 Exploratory Data Analysis')

RackioAI.append_data(EDA1)

RackioAI.append_data(EDA2)


print(RackioAI.summary())