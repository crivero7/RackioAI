from rackio_AI import RackioAI, Preprocessing
from rackio import Rackio

app = Rackio()

RackioAI(app)

preprocess1 = Preprocessing(name='Preprocess1', description='preprocess for data', problem_type='regression')

preprocess2 = Preprocessing(name='Preprocess2', description='preprocess for data', problem_type='classification')

RackioAI.append_preprocessing_model(preprocess1)

RackioAI.append_preprocessing_model(preprocess2)


print(RackioAI.summary())
