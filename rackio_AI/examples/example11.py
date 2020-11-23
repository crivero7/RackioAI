from rackio_AI import RackioEDA, RackioAI, Preprocessing
from rackio import Rackio

app = Rackio()

RackioAI(app)

EDA1 = RackioEDA(name='EDA1', description='Object 1 Exploratory Data Analysis')

EDA2 = RackioEDA(name='EDA2', description='Object 2 Exploratory Data Analysis')

RackioAI.append_data(EDA1)

RackioAI.append_data(EDA2)

preprocess1 = Preprocessing(name='Preprocess1', description='preprocess for data', problem_type='regression')

preprocess2 = Preprocessing(name='Preprocess2', description='preprocess for data', problem_type='classification')

RackioAI.append_preprocessing_model(preprocess1)

RackioAI.append_preprocessing_model(preprocess2)


print(RackioAI.summary())
