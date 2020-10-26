import pandas as pd
import numpy as np
from rackio_AI.core import RackioAI
from rackio_AI.preprocessing import Scaler
from rackio_AI.preprocessing import KalmanFilter


class Preprocessing:
    """
    This class allows to you do preprocessing to the data in *RackioAI* or *RackioEDA
    """

    app = RackioAI()

    def __init__(self, name, description, problem_type='regression'):
        """
        Preprocessing instantiation

        ___
        **Parameters**

        * **:param name:** (str) Preprocessing model name
        * **:param description:** (str) Preprocessing model description
        * **:param problem_type:** (str) Preprocessing model type
            * *regression*
            * *classification*

        **:return:**

        * **preprocessing:** (Preprocessing object)

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')

        ```
        """
        self._data = None
        self._name = name
        self._description = description
        self._type = problem_type

        if problem_type.lower() in ['regression', 'classification']:

            if problem_type.lower() == 'regression':

                self.preprocess = Regression(name, description)

            else:

                self.preprocess = Classification(name, description)

        self.scaler = Scaler()
        self.kalman_filter = KalmanFilter()

    @property
    def data(self):
        """
        Preprocessing attribute to storage data values

        **Parameters**

        * **:param value:** (pandas.DataFrame or np.ndarray)

        * **:return:**

        * **data:** (pandas.DataFrame)
        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Preprocessing attribute to storage data values

        **Parameters**

        * **:param value:** (pandas.DataFrame or np.ndarray)

        * **:return:**

        * **data:** (pandas.DataFrame)
        """
        if isinstance(value, pd.DataFrame) or isinstance(value, np.ndarray):

            if isinstance(value, np.ndarray):
                value = pd.DataFrame(value)

            self.synthetic_data.data = value
        else:
            raise TypeError('value must be a pd.DataFrame or np.ndarray')

        self._data = value

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
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name='Preprocess model name', description='preprocess for data', problem_type='regression')
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

    def __call__(self, action):
        """
        Invoker to apply any preprocessing action

        **Parameters**

        * **:param action:** (str) preprocessing to do
            * *scaler*
            * *filter*
            * *split*

        **:return:**

        * **data:** (pd.DataFrame) Preprocessed data

        """
        allowed_actions = ["scaler", "filter", "split"]

        if action.lower() in allowed_actions:

            todo = getattr(self.preprocess,action.lower())

            data = todo(self.data)

            return data
        else:
            raise NotImplementedError("{} method is not implemented in {} class".format(action, self._preprocesssing.get_type().get_name()))

        return

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
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name='Preprocess model name', description='preprocess for data', problem_type='regression')
        >>> preprocess.serialize()
        {'name': 'Preprocess model name', 'description': 'preprocess for data', 'type': 'regression'}

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
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name='Preprocess model name', description='preprocess for data', problem_type='regression')
        >>> preprocess.get_name()
        'Preprocess model name'

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