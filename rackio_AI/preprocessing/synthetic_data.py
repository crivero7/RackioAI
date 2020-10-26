import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rackio_AI.decorators import typeCheckedAttribute
from rackio_AI.preprocessing.synthetic_data_base import PrepareData


data_type_synthetic_data = {'_data': [pd.Series, pd.DataFrame, np.ndarray],
                            'error': [np.ndarray],
                            'repeteability': [np.ndarray],
                            'lower_limit': [np.ndarray],
                            'upper_limit': [np.ndarray],
                            'dead_band': [np.ndarray]}

@typeCheckedAttribute.typeassert(**data_type_synthetic_data)
class SyntheticData(PrepareData):
    """
    This class allows you to add anomalies to a data to model the behavior of a data that comes from the field
    You can add the following anomalies:

    * **Gaussian noise**
    * **Outliers**
    * **Frozen data**
    * **Excessive noise**
    * **Out of range data**
    * **Instrument decalibration**
    * **Sensor drift**
    """
    def __init__(self):
        """

        """
        super(SyntheticData, self).__init__()
        self._data = np.array([])
        options = {'error': np.array([]),
                   'repeteability': np.array([]),
                   'lower_limit': np.array([]),
                   'upper_limit': np.array([]),
                   'dead_band': np.array([])}
        options = self._check_options(**options)
        {setattr(self, key, options[key]) for key in options.keys()}
        self.accuracy = self.error - self.repeteability
        self.span = self.upper_limit - self.lower_limit

    @property
    def data(self):
        """
        Property method

        @setter
        **Parameters**

        * **:param value:** (np.ndarray or pandas.DataFrame)

        **:return:**

        * **data:** (np.array, pd.DataFrame)
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Property setter methods

        **Parameters**

        * **value:** (np.array, pd.DataFrame)

        **return**

            None
        """
        if not isinstance(value, np.ndarray):
            self._data = value.values
        else:
            self._data = value

    def set_options(self, **options):
        """
        This method allows to you set user options like instrument attributes, if the option is not passed by the user,
        then this function set it zeros

        **Parameters**

        * **:param options:**
            * **error:** (np.ndarray, list)
            * **repeteability:** (np.ndarray, list)
            * **lower_limit:** (np.ndarray, list)
            * **upper_limit:** (np.ndarray, list)
            * **dead_band:** (np.ndarray, list)

        **:return:**

        None
        """
        options = self._check_options(**options)
        # convert the options to np.array
        for key in options.keys():
            if isinstance(options[key], np.ndarray):
                setattr(self, key, options[key])
            elif isinstance(options[key], list):
                setattr(self, key, np.array(options[key]))
        # accuracy and span computations
        self.accuracy = self.error - self.repeteability
        self.span = self.upper_limit - self.lower_limit

    @PrepareData.step
    def add_instrument_error(self):
        """
         Add insturment error according the error and repeteability instrument

        **Parameters**

            None

        **:return:**

        * **data:** (np.ndarray) data with instrument error
        """
        # Adding sensibility according to the deadband instrument
        self._add_dead_band()
        # Adding instrument error to all data
        self.data += (self.repeteability * (2*np.random.random(self.data.shape)-1) + self.accuracy*(2*np.random.random(self.data.shape)-1))* self.span
        return self.data

    @PrepareData.step
    def add_decalibration(self, decalibration_factor=2.0, duration=10):
        """
        Add instrument decalibration to the data

        **Parameters**

        * **decalibration_factor:** (float) default=2.0: Instrument error amplitude respect to original instrument error
        * **duration:** (int) default=10: Values followed in the data in which the anomaly will be maintained

        **return**

        * **data:** (np.ndarray) data with instrument decalibration
        """
        bias = -decalibration_factor * self.error - self.repeteability
        init_position = np.random.randint(self.data.shape[0]-duration, size=(1,self.data.shape[-1]))[0]
        decal = (self.repeteability * np.random.random([duration, self.data.shape[-1]]) - bias) * self.span
        for count, pos in enumerate(init_position):
            min = bool(np.random.randint(2))
            if min:
                self.data[pos:pos + duration, count] -= decal[:, count]
            else:
                self.data[pos:pos+duration, count] += decal[:, count]

        return self.data

    @PrepareData.step
    def add_sensor_drift(self, sensor_drift_factor=5.0, duration=10):
        """
        Adding sensor drift anomaly to the data

        **Parameters**

        * **sensor_drift_factor:** (float) default=5.0: Instrument sensor drift amplitude respect to original instrument error
        * **duration:** (int) default=10: Values followed in the data in which the anomaly will be maintained

        **return**

        * **data:** (np.ndarray) data with sensor drift
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        # initial drift value
        drift = np.array([self.data[pos, count] for count, pos in enumerate(init_position)])
        drift = drift.reshape((1,self.data.shape[-1]))
        # Adding sensor drift to the data
        for k in range(duration-1):
            min = bool(np.random.randint(2))
            if min:
                new_value = drift[k, :] - sensor_drift_factor * self.error * self.span / duration
            else:
                new_value = drift[k,:] + sensor_drift_factor * self.error * self.span / duration

            drift = np.append(drift, new_value.reshape([1, self.data.shape[-1]]), axis=0)
        for count, pos in enumerate(init_position):
            self.data[pos:pos+duration, count] = drift[:,count]

        return self.data

    @PrepareData.step
    def add_excessive_noise(self, error_factor=3.0, repeteability_factor=3.0, duration=10):
        """
        Adding excessive gaussian noise anomaly to the data

        **Parameters**

        * **error_factor:** (float) default=3.0: Instrument error amplitude respect to original instrument error
        * **repeteability_factor:** (float) default=3.0: Instrument repeteability amplitude respect to original repeteability
        * **duration:** (int) default=10: Values followed in the data in which the anomaly will be maintained

        **return**

        * **data:** (np.ndarray) data with excessive white noise
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        repeteability = repeteability_factor * self.repeteability
        error = error_factor * self.error
        accuracy = error - repeteability
        new_value = (repeteability * (2 * np.random.random([duration, self.data.shape[-1]]) - 1) +
                    accuracy * (2 * np.random.random([duration, self.data.shape[-1]]) - 1)) * self.span
        for count, pos in enumerate(init_position):
            self.data[pos:pos+duration, count] +=  new_value[:,count]

        return self.data

    @PrepareData.step
    def add_frozen_data(self, duration=10):
        """
        Adding frozing anomaly to the data

        **Parameters**

        * **duration:** (int) default=10: Values followed in the data in which the anomaly will be maintained

        **return**

        * **data:** (np.ndarray) data with frozen instrument
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        for count, pos in enumerate(init_position):
            self.data[pos:pos+duration, count] = self.data[pos, count]

        return self.data

    @PrepareData.step
    def add_outlier(self, span_factor=0.03):
        """
        Adding outlier anomaly to the data

        **Parameters**

        * **span_factor:** (float) default=0.03: Fraction respect to instrument range to add to the data

        **return**

        * **data:** (np.ndarray) data with outlier
        """
        init_position = np.random.randint(self.data.shape[0], size=(1, self.data.shape[-1]))[0]
        outlier = span_factor * self.span
        for count, pos in enumerate(init_position):
            min = bool(np.random.randint(2))
            if min:
                self.data[pos, count] -= outlier[count]
            else:
                self.data[pos, count] += outlier[count]

        return self.data

    @PrepareData.step
    def add_out_of_range(self, duration=10):
        """
        Adding out of range anomaly to the data

        **Parameters**

        * **duration:** (int) default=10: Values followed in the data in which the anomaly will be maintained

        **return**

        * **data:** (np.ndarray) data with out of range
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        anomaly = 0.07 * self.span
        for count, pos in enumerate(init_position):
            min = bool(np.random.randint(2))
            if min:
                self.data[pos:pos+duration, count] = self.lower_limit[count] - anomaly[count]
            else:
                self.data[pos:pos+duration, count] = self.upper_limit[count] + anomaly[count]

        return self.data

    def done(self, view=False, **options):
        """
        This method allows to you plot the anomalies added to the data

        **Parameters**

        * **view:** (bool) default=False: If False no plot, True plot
        * **options:**
            * **columns:** (list[int]) columns to plot in a list
            * **ylabel:** (str) ylabel string
            * **xlabel:** (str) xlabel string

        **return**

            None
        """
        if view:
            self.view(columns=options['columns'], ylabel=options['ylabel'], xlabel=options['ylabel'])

    def __call__(self, decalibrations=0, sensor_drift=0, excesive_noise=0, frozen_data=0, outliers=0, out_of_range=0, add_WN=False, **options):
        """
        Callback to do anomalies

        ___
        **Parameters**

        * **decalibrations:** (int) default=0: decalibration anomalies to add
        * **sensor_drift:** (int) default=0: sensor drift anomalies to add
        * **excesive_noise:** (int) default=0: excesive noise anomalies to add
        * **frozen_data:** (int) default=0: frozen data anomalies to add
        * **outliers:** (int) default=0: outlier anomalies to add
        * **out_of_range:** (int) default=0: out of range anomalies to add
        * **add_WN:** (bool) default=False: add or not add error instrumentation
        * **options:**
            * **duration:** (dict)
                * **min:** (int) default=10
                * **max:** (int) default=50
            * **view:** (bool) default=False
            * **columns:** (list[int]) default=[0]

        **return:**

        * **data:** (np.array, pd.DataFrame): data with anomalies
        ___

        ## Snippet code

        ```python
        >>> import os
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> os.chdir('..')
        >>> cwd = os.getcwd()
        >>> filename = os.path.join(cwd,'data','pkl_files', 'test_data.pkl')
        >>> RackioAI.load(filename)
                    Pipe-60 Totalmassflow_(KG/S)  ...  Pipe-151 Pressure_(PA)
        0.000000                        37.83052  ...                352683.3
        0.502732                        37.83918  ...                353449.8
        1.232772                        37.83237  ...                353587.3
        1.653696                        37.80707  ...                353654.8
        2.200430                        37.76957  ...                353706.8
        ...                                  ...  ...                     ...
        383.031800                     169.36700  ...                374582.2
        383.518200                     169.37650  ...                374575.9
        384.004500                     169.38550  ...                374572.7
        384.490900                     169.39400  ...                374573.0
        384.977200                     169.40170  ...                374576.1
        <BLANKLINE>
        [20000 rows x 4 columns]
        >>> variable_names = RackioAI.data.columns.to_list()
        >>> error = [0.0025, 0.0025, 0.0025, 0.0025]
        >>> repeteability = [0.001, 0.001, 0.001, 0.001]
        >>> lower_limit = [0, 0, 400000, 100000]
        >>> upper_limit = [500, 500, 1200000, 600000]
        >>> dead_band = [0.001, 0.001, 0.001, 0.001]
        >>> RackioAI.synthetic_data.set_options(error=error, repeteability=repeteability, lower_limit=lower_limit, upper_limit=upper_limit, dead_band=dead_band)
        >>> data = RackioAI.synthetic_data(frozen_data=2, out_of_range=1, add_WN=True, view=False, columns=[0,1,2,3], duration={'min': 20, 'max': 100})

        ```
        """
        default_options = {'duration': {'min': 10,
                                        'max': 50},
                           'view': False,
                           'columns': [0]}

        options = {key: options[key] if key in options.keys() else default_options[key] for key in default_options}

        duration_min = options['duration']['min']
        duration_max = options['duration']['max']

        # Adding decalibration
        for i in range(decalibrations):
            duration = np.random.randint(duration_min, duration_max)
            self.add_decalibration(duration=duration)

        # Adding sensor drift
        for i in range(sensor_drift):
            duration = np.random.randint(duration_min, duration_max)
            self.add_sensor_drift(duration=duration)

        # Adding instrument error
        if add_WN:
            self.add_instrument_error()

        # Adding excessive gaussian noise
        for i in range(excesive_noise):
            duration = np.random.randint(duration_min, duration_max)
            self.add_excessive_noise(duration=duration)

        # Adding frozen data
        for i in range(frozen_data):
            duration = np.random.randint(duration_min, duration_max)
            self.add_frozen_data(duration=duration)

        # Adding outliers
        for i in range(outliers):
            span_factor = np.random.randint(11) / 100
            self.add_outlier(span_factor=span_factor)

        # Adding out of range data
        for i in range(out_of_range):
            duration = np.random.randint(duration_min, duration_max)
            self.add_out_of_range(duration=duration)

        self.done(view=options['view'], columns=options['columns'], ylabel='Amplitude', xlabel='Point')
        return self.data

    def round_by_dead_band(self, data):
        """
        Round data according to the instrument dead band

        **Parameters**

        * **data:** (np.array)

        **return**

        * **data:** (np.array): round data applied
        """
        return np.array([np.round(data[:, count] * (10 **str(value)[::-1].find('.'))) / (10 **str(value)[::-1].find('.'))
                for count, value in enumerate(self.dead_band)])

    @typeCheckedAttribute.checkOptions
    def _check_options(self, **options):
        """
        This method allows to you check user options, if the option is not passed by the user, then this function set it
        zeros

        **Paramters**

        * **options:** {'error': [np.ndarray, list],
                          'repeteability': [np.ndarray, list],
                          'lower_limit': [np.ndarray, list],
                          'upper_limit': [np.ndarray, list],
                          'dead_band': [np.ndarray, list]}

        **return**

        * **options:** (dict) valid options
        """
        if isinstance(self._data, pd.Series):
            sensors_numbers = 1
        else:
            sensors_numbers = self._data.shape[-1]
        # Default parameters instrumentation definition
        default_options = {key:np.zeros(sensors_numbers) for key in data_type_synthetic_data if key != '_data'}

        options = {key: options[key] if key in options.keys() else default_options[key] for key in default_options}

        return options

    def _add_dead_band(self):
        """
        This method allows to you to add instrument sensibility behavior to the data

        **Parameters**

            None

        **return**

        * **data:** (np.ndarray) round data
        """
        data = self.data
        difference = np.diff(data, axis=0)
        # Positions where the new value has reached the instrument sensibility
        pos_true = abs(difference) >= self.dead_band
        # Positions where the new value has not changed enough
        pos_false = abs(difference) < self.dead_band
        # decimals to be rounded
        decimals = [str(self.dead_band[count])[::-1].find('.') for count in range(self.dead_band.shape[-1])]
        # Applying round
        for count in range(pos_true.shape[-1]):
            data[1::,:][pos_false[:,count], count] = np.round(data[:-1:,:][pos_false[:,count], count], decimals[count])
            data[1::,:][pos_true[:,count],count] = np.round(data[1::,:][pos_true[:,count],count], decimals[count])

        self.data = data

        return data

    def view(self, columns=[0], xlabel='Time', ylabel='Amplitude'):
        """
        Plot the data with anomalies added

        **Paramters**

        * **columns:** (list) default=[0]:
        * **xlabel:** (str) default='Time'
        * **ylabel:** (str) default='Amplitude'

        **return**

            None
        """
        plt.plot(self.data[:, columns])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show(block=True)

if __name__=="__main__":
    import doctest
    doctest.testmod()