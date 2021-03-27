import pandas as pd
import numpy as np
from rackio_AI.core import RackioAI
from rackio_AI.preprocessing import RackioAIScaler, RackioAISplitter, KalmanFilter, LSTMDataPreparation
from rackio_AI.preprocessing import SyntheticData, RackioAIFE


class Preprocessing:
    """
    This class allows to you do preprocessing to the data in *RackioAI* or *RackioEDA

    ```python
    >>> from rackio_AI import Preprocessing
    >>> preprocessing = Preprocessing(name='Preprocessing core', description='preprocessing for data', problem_type='regression')
    
    ```
    
    """
    scaler = RackioAIScaler()
    splitter = RackioAISplitter()
    kalman_filter = KalmanFilter()
    lstm_data_preparation = LSTMDataPreparation()
    synthetic_data = SyntheticData()
    feature_extraction = RackioAIFE()
    app = RackioAI()

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
        self.app.append(self)

        if problem_type.lower() in ['regression', 'classification']:

            if problem_type.lower() == 'regression':

                self.preprocess = Regression(name, description)

            else:

                self.preprocess = Classification(name, description)

    @property
    def data(self):
        """
        Preprocessing attribute to storage data values

        **Parameters**

        * **:param value:** (pandas.DataFrame or np.ndarray)

        * **:return:**

        * **data:** (pandas.DataFrame)
        """
        return self.app.data

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
            
            self.app.data = pd.DataFrame(value)
        
        else:
            
            self.app.data = value

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
        >>> from rackio_AI import RackioAI
        >>> preprocessing = RackioAI.get(name='Preprocessing core', _type='Preprocessing')
        >>> preprocessing.description
        'preprocessing for data'

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
        >>> from rackio_AI import RackioAI
        >>> preprocessing = RackioAI.get(name='Preprocessing core', _type='Preprocessing')
        >>> preprocessing.serialize()
        {'name': 'Preprocessing core', 'description': 'preprocessing for data', 'type': 'regression'}

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
        >>> from rackio_AI import RackioAI
        >>> preprocessing = RackioAI.get(name='Preprocessing core', _type='Preprocessing')
        >>> preprocessing.get_name()
        'Preprocessing core'

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