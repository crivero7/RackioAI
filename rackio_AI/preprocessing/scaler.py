from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import Normalizer, Binarizer, QuantileTransformer, PowerTransformer
import pandas as pd

class RackioAIScaler:
    """
    This is a *RackioAI* preprocessing class to scale the data to create a Deep learning model
    ___

    ## **Scaling**

    ### **Preprocessing Instantiation**

    ### **min_max scaler**

    ```python
    >>> from rackio_AI import RackioAI
    >>> preprocessing = RackioAI.get("Preprocessing", _type="Preprocessing")
    >>> min_max_scaler = preprocessing.scaler.min_max()
    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
    >>> min_max_scaler.fit_transform(df)
    array([[0. , 0. , 0. ],
           [0.5, 0.5, 0.5],
           [1. , 1. , 1. ]])
    
    ```

    ### **min_max inverse transform**

    ```python 
    >>> df_scaled = min_max_scaler.fit_transform(df)
    >>> min_max_scaler.inverse_transform(df_scaled)
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    
    ```

    ### **standard scaler**

    ```python
    >>> standard_scaler = preprocessing.scaler.standard()
    >>> standard_scaler.fit_transform(df)
    array([[-1.22474487, -1.22474487, -1.22474487],
           [ 0.        ,  0.        ,  0.        ],
           [ 1.22474487,  1.22474487,  1.22474487]])

    ```
    ### **standard inverse transform**

    ```python 
    >>> df_scaled = standard_scaler.fit_transform(df)
    >>> standard_scaler.inverse_transform(df_scaled)
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    
    ```
    """
    min_max = MinMaxScaler
    standard = StandardScaler
    max_abs = MaxAbsScaler
    robust = RobustScaler
    normalizer = Normalizer
    binarizer = Binarizer
    quantile_transform = QuantileTransformer
    power_transform = PowerTransformer

    def __str__(self):
        """

        :return:
        """
        return "Scaler Object\n{}".format(self.__name__)
