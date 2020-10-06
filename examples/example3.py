from rackio_AI.preprocess import Preprocess
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

preprocess1 = Preprocess(name= 'Preprocess1',description='preprocess for data', problem_type='regression')

preprocess2 = Preprocess(name= 'Preprocess2',description='preprocess for data', problem_type='classification')

RackioAI.append_preprocess_model(preprocess1)

RackioAI.append_preprocess_model(preprocess2)


print(RackioAI.summary())