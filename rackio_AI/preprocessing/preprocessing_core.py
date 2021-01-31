import pandas as pd
import numpy as np
from rackio_AI.core import RackioAI
from rackio_AI.preprocessing import RackioAIScaler, Splitter, KalmanFilter, LSTMDataPreparation


app = RackioAI()

class Preprocessing:
    """
    This class allows to you do preprocessing to the data in *RackioAI* or *RackioEDA
    
    """
    scaler = RackioAIScaler()

    def __init__(
        self, 
        name: str='', 
        description: str='', 
        problem_type: str='regression'
        ):
        """
        Preprocessing instantiation

        """
        self._data = None
        self._name = name
        self._description = description
        self._type = problem_type
        app.append(self)

        if problem_type.lower() in ['regression', 'classification']:

            if problem_type.lower() == 'regression':

                self.preprocess = Regression(name, description)

            else:

                self.preprocess = Classification(name, description)

        self.kalman_filter = KalmanFilter()
        self.splitter = Splitter()
        self.lstm_data_preparation = LSTMDataPreparation()


    @property
    def data(self):
        """
        Preprocessing attribute to storage data values

        **Parameters**

        * **:param value:** (pandas.DataFrame or np.ndarray)

        * **:return:**

        * **data:** (pandas.DataFrame)
        """
        return app.data

    @data.setter
    def data(self, value):
        """
        Preprocessing attribute to storage data values

        **Parameters**

        * **:param value:** (pandas.DataFrame or np.ndarray)

        * **:return:**

        * **data:** (pandas.DataFrame)
        """
        if isinstance(value, np.ndarray):
            
            app.data = pd.DataFrame(value)
        
        else:
            
            app.data = value

    @property
    def description(self):
        """
        Preprocessing attribute to storage preprocessing model description

        ___
        **Parameters**

        * **:param value:** (str) Preprocessing model description

        * **:return:**

        * **description:** (str) Preprocessing model description

        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import Preprocessing
        >>> preprocess = Preprocessing(name='Preprocess 2', description='preprocess for data', problem_type='regression')
        >>> preprocess.description
        'preprocess for data'

        ```
        """
        return self._description

    @description.setter
    def description(self, value):
        """

        """
        self._description = value

    def serialize(self):
        """
        Serialize preprocessing object

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (dict) keys {"name", "description", "type"}

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import Preprocessing
        >>> preprocess = Preprocessing(name='Preprocess 3', description='preprocess for data', problem_type='regression')
        >>> preprocess.serialize()
        {'name': 'Preprocess 3', 'description': 'preprocess for data', 'type': 'regression'}

        ```
        """
        result = {"name": self.get_name(),
                  "description": self.description,
                  "type": self._type}

        return result

    def get_name(self):
        """
        Get preprocessing model name

        ___
        **Parameters**

        None

        **:return:**

        * **name:** (str) Preprocessing name

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> preprocess = Preprocessing(name='Preprocess 4', description='preprocess for data', problem_type='regression')
        >>> preprocess.get_name()
        'Preprocess 4'

        ```
        """

        return self._name


class Regression(Preprocessing):
    """
    This class contains preproccesing action allowed for **Regression** problems

    """

    def __init__(self, name, description):
        """
        Initializer Regression object

        **Parameters**

        * **:param name:** (str) Preprocessing  model's name for regression problem
        * **:param description:** (str) Preprocessing model's description for regression problem
        """
        self._name = name
        self._description = description


class Classification(Preprocessing):
    """
    This class contains preproccesing action allowed for **Classification** problems

    """

    def __init__(self, name, description):
        """
        Initializer Classification object

        **Parameters**

        * **:param name:** (str) Preprocessing model's name for classification problem
        * **:param description:** (str) Preprocessing model's description for classification problem
        """
        self._name = name
        self._description = description

if __name__=="__main__":
    
    import doctest

    doctest.testmod()