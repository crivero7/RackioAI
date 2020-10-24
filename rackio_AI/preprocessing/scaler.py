from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd



class Scaler:

    def __init__(self, _type='minmax', **kwargs):
        """
        _type (str): 'minmax' or 'standard'
        kwargs: {'range': tuple (0,1) if _type is 'minmax', else no kwargs}
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

    def __call__(self, data):
        """
        data: np.array or pd.dataframe

        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        return self._scaler.fit(data)

    def __str__(self):
        """

        """
        pass

    def apply_inverse(self, data):
        """
        data: np.array or pd.dataframe

        """
        return self._scaler.inverse_transform(data)

    @property
    def range(self):

        return self._range

    @range.setter
    def range(self, value=(0, 1)):
        """
        value (tuple): (min,max) values
        """
        if isinstance(self._scaler, MinMaxScaler):
            self._scaler = MinMaxScaler(feature_range=value)

        self._range = value