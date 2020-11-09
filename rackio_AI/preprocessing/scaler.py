from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


class Scaler:
    """
    This is a *RackioAI* preprocessing class to scale the data to create a Deep learning model
    """

    def __init__(self, _type='minmax', **kwargs):
        """
        Scaler instantiation

        **Parameters**

        * **:param _type:**  'minmax' or 'standard'
        * **:param kwargs:** {'range': tuple (0,1) if _type is 'minmax', else no kwargs}

        **:return:**

        * **scaler:** (Scaler object)

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')
        >>> print(preprocess.scaler)
        Scaler Object
        MinMaxScaler()

        ```
        """

        if _type.lower() in ['minmax', 'standard']:

            if _type.lower() == 'minmax':

                kwargs_default = {'range': (0, 1)}
                options = {key: kwargs[key] if key in kwargs.keys() else kwargs_default[key] for key in
                           kwargs_default.keys()}
                self._range = options['range']
                self._scaler = MinMaxScaler(feature_range=self._range)

            else:

                self._scaler = StandardScaler()
                delattr(self, 'range')

        else:
            raise TypeError('scaler {} is not available in class {}'.format(_type, self.__class__.__name__))

    @property
    def range(self):
        """
        Property method

        **Parameters**

        * **value (tuple):** (min,max) values

        **:return:**

        * **value (tuple):** (min,max) values
        """
        return self._range

    @range.setter
    def range(self, value=(0, 1)):
        """
        value (tuple): (min,max) values
        """
        if isinstance(self._scaler, MinMaxScaler):
            self._scaler = MinMaxScaler(feature_range=value)

        self._range = value

    def __call__(self, data):
        """
        This is the callable method to execute the data scaling

        **Parameters**

        * **:param data:** (np.array or pd.dataframe)

        **:return:**

        * **data:** (np.array or DataFrame) scaled values

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> import numpy as np
        >>> import pandas as pd
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')
        >>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> # MinMaxScaler
        >>> preprocess.scaler(df)
        array([[0. , 0. , 0. ],
               [0.5, 0.5, 0.5],
               [1. , 1. , 1. ]])
        >>> # StandardScaler
        >>> preprocess.scaler.set_options(_type='standard')
        >>> preprocess.scaler(df)
        array([[-1.22474487, -1.22474487, -1.22474487],
               [ 0.        ,  0.        ,  0.        ],
               [ 1.22474487,  1.22474487,  1.22474487]])

        ```
        """

        if isinstance(data, pd.DataFrame):
            data = data.values

        return self._scaler.fit_transform(data)

    def __str__(self):
        """

        :return:
        """
        return "Scaler Object\n{}".format(self._scaler)

    def get_inverse(self, data):
        """
        Get the inverse scaling

        **Paramerters**

        * **:param data:** (np.array or pd.DataFrame)

        **:return:**

        * **data:** (np.arrray or pd.DataFrame)

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> import numpy as np
        >>> import pandas as pd
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')
        >>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> data = preprocess.scaler(df)
        >>> preprocess.scaler.get_inverse(data)
        array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 9.]])

        ```
        """

        return self._scaler.inverse_transform(data)

    def set_options(self, _type='minmax', **kwargs):
        """
        set options to Scaler object

        **Parameters**

        * **:param _type:**  'minmax' or 'standard'
        * **:param kwargs:** {'range': tuple (0,1) if _type is 'minmax', else no kwargs}

        **:return:**

        * **scaler:** (Scaler object)

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')
        >>> # MinMaxScaler
        >>> preprocess.scaler.set_options(_type='minmax', range=(-1,1))
        >>> print(preprocess.scaler)
        Scaler Object
        MinMaxScaler(feature_range=(-1, 1))
        >>> # StandardScaler
        >>> preprocess.scaler.set_options(_type='standard')
        >>> print(preprocess.scaler)
        Scaler Object
        StandardScaler()

        ```
        """
        if _type.lower() in ['minmax', 'standard']:

            if _type.lower() == 'minmax':

                kwargs_default = {'range': (0, 1)}
                options = {key: kwargs[key] if key in kwargs.keys() else kwargs_default[key] for key in kwargs_default.keys()}
                self._range = options['range']
                self._scaler = MinMaxScaler(feature_range=self._range)

            else:

                self._scaler = StandardScaler()

                if hasattr(self, '_range'):

                    delattr(self, '_range')

        else:

            raise TypeError('scaler {} is not available in class {}'.format(_type, self.__class__.__name__))

if __name__=="__main__":
    import doctest
    doctest.testmod()