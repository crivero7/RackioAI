from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import Normalizer, Binarizer, QuantileTransformer, PowerTransformer
from rackio_AI.utils.utils_core import Utils
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
    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
    >>> preprocessing.scaler(df, method='min_max')
       One  Two  Three
    0  0.0  0.0    0.0
    1  0.5  0.5    0.5
    2  1.0  1.0    1.0
    
    ```

    ### **min_max inverse transform**

    ```python 
    >>> df_scaled = preprocessing.scaler(df, method='min_max')
    >>> preprocessing.scaler.inverse(df_scaled)
       One  Two  Three
    0  1.0  2.0    3.0
    1  4.0  5.0    6.0
    2  7.0  8.0    9.0
    
    ```

    ### **standard scaler**

    ```python
    >>> preprocessing.scaler(df, method='standard')
            One       Two     Three
    0 -1.224745 -1.224745 -1.224745
    1  0.000000  0.000000  0.000000
    2  1.224745  1.224745  1.224745

    ```
    ### **standard inverse transform**

    ```python 
    >>> df_scaled = preprocessing.scaler(df, method='min_max')
    >>> preprocessing.scaler.inverse(df_scaled)
       One  Two  Three
    0  1.0  2.0    3.0
    1  4.0  5.0    6.0
    2  7.0  8.0    9.0
    
    ```
    """
    methods = {
        "min_max": MinMaxScaler,
        "standard": StandardScaler,
        "max_abs": MaxAbsScaler,
        "robust": RobustScaler,
        "normalizer": Normalizer,
        "binarizer": Binarizer,
        "quantile_transform": QuantileTransformer,
        "power_transform": PowerTransformer
    }

    def __init__(self):
       """Documentation here"""
       self.__scaler = None

    def __call__(self, df, method: str="min_max", columns: list=[], **kwargs):
        """Documentation here"""
        if not method.lower() in self.methods:
            
            raise TypeError("{} method not available, availables methods: {}".format(method, methods.keys()))

        self.__scaler = self.methods[method.lower()](**kwargs)
        column_name = Utils.get_column_names(df)

        return pd.DataFrame(self.__scaler.fit_transform(df), columns=column_name)

    def inverse(self, df, columns: list=[]):
        """Documentation here"""
        column_name = Utils.get_column_names(df)
        return pd.DataFrame(self.__scaler.inverse_transform(df), columns=column_name)


if __name__=='__main__':

    import doctest

    doctest.testmod()