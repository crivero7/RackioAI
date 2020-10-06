from rackio_AI.decorators import typeCheckedAttribute
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .synthetic_data_base import PrepareData


data_type_synthetic_data = {'_data': [pd.Series, pd.DataFrame, np.ndarray],
                         'error': [np.ndarray],
                         'repeteability': [np.ndarray],
                         'lower_limit': [np.ndarray],
                         'upper_limit': [np.ndarray],
                         'dead_band': [np.ndarray]}

@typeCheckedAttribute.typeassert(**data_type_synthetic_data)
class SyntheticData(PrepareData):
    """
    Esta clase permite agregarle anomalías a una data de una simulación para modelar el comportamiento de una data
    que proviene de campo; es decir, por medio de esta clase, se puede agregar las siguientes anomalías que podrían tener
    los instrumentos de campo:
    Ruido Gaussiano
    Outliers
    Data Congelada
    Ruido Excesivo
    Dato Fuera de Rango
    Descalibración de Instrumentos
    Deriva del Sensor

    Ejemplo
            Sensor1  Sensor 2 Sensor3 ... Sensor n
    data = [[215.5,   321.5,   322,  ..., 225],          # tiempo 1
            [215.5,   321.5,   322,  ..., 225],          # tiempo 2
            [215.5,   321.5,   322,  ..., 225],          # tiempo 3
            [  . ,      .,      .,  ..., .  ],           # .
            [  . ,      .,      .,  ..., .  ],           # .
            [215.5,   321.5,   322,  ..., 225]]          # tiempo n
    error = [E1, E2, E3, ... En]
    repeteability = [Rep1, Rep2, Rep3, ... Repn]
    lower_limit = [LL1, LL2, LL3, ... LLn]
    uppe_limit = [UL1, UL2, UL3, ... ULn]
    dead_band = [DB1, DB2, DB3, ... DBn]
    obj = SyntheticData(data, error=error, repeteability=repeteability, lower_limit=lower_limit, upper_limit=upper_limit, dead_band=dead_band)
    """
    _data = np.array([])
    error = np.array([])
    repeteability = np.array([])
    lower_limit = np.array([])
    upper_limit = np.array([])
    dead_band = np.array([])


    def __init__(self, **options):

        super(SyntheticData, self).__init__()
        self._data = np.zeros(0)
        options = self._check_options(**options)
        {setattr(self, key, options[key]) for key in options.keys()}
        self.accuracy = self.error - self.repeteability
        self.span = self.upper_limit - self.lower_limit


    @property
    def data(self):
        """

        """
        return self._data

    @data.setter
    def data(self, value):
        """

        """
        if not isinstance(value, np.ndarray):
            self._data = value.values
        else:
            self._data = value

    def set_options(self, **options):
        """

        """
        options = self._check_options(**options)
        {setattr(self, key, options[key]) for key in options.keys()}
        self.accuracy = self.error - self.repeteability
        self.span = self.upper_limit - self.lower_limit

    @PrepareData.step
    def add_instrument_error(self):
        self._add_dead_band()
        self.data += (self.repeteability * (2*np.random.random(self.data.shape)-1) + self.accuracy*(2*np.random.random(self.data.shape)-1))* self.span

    @PrepareData.step
    def add_decalibration(self, decalibration_factor=2, duration=10):
        """
        Este método permite agregar el fenomeno de descalibración a una data simulada

        min: Booleano que permite establecer si el outlier es positivo (False) o negativo (True)
        decalibrationFactor: Es la magnitud que multiplica al error del instrumento para generar la descalibración
        duration: es la cantidad de valores seguidos en la data en la que se mantenddrá la anomalía
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

    @PrepareData.step
    def add_sensor_drift(self, sensor_drift_factor=5, duration=10):
        """
        Este método permite agregarle deriva del sensor a la data.

        min: Booleano que permite establecer si el outlier es positivo (False) o negativo (True)
        decalibrationFactor: Es la magnitud que multiplica al error del instrumento para generar el valor final al que
        llegará el fenómeno de deriva del sensor.
        duration: es la cantidad de valores seguidos en la data en la que se mantenddrá la anomalía
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        "initial drift value"
        drift = np.array([self.data[pos, count] for count, pos in enumerate(init_position)])
        drift = drift.reshape((1,self.data.shape[-1]))
        "Agregando el fenomeno de deriva del sensor"
        for k in range(duration-1):
            min = bool(np.random.randint(2))
            if min:
                new_value = drift[k, :] - sensor_drift_factor * self.error * self.span / duration
                drift = np.append(drift, new_value.reshape([1, self.data.shape[-1]]), axis=0)
            else:
                new_value = drift[k,:] + sensor_drift_factor * self.error * self.span / duration
                drift = np.append(drift, new_value.reshape([1, self.data.shape[-1]]), axis=0)
        "Agregando la deriva a la data del instrumento"
        for count, pos in enumerate(init_position):
            self.data[pos:pos+duration, count] = drift[:,count]

    @PrepareData.step
    def add_excesive_noise(self, error_factor=3, repeteability_factor=3, duration=10):
        """
        Este método permite agregarle ruido blanco excesivo

        errorFactor: Factor de multiplicación para aumentar el ruido del instrumento
        repeteabilityFactor: Factor de multiplicación para aumentar la repetibilidad del instrumento.
        duration: Cantidad de valores seguidos en la data en la que se mantenddrá la anomalía
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        repeteability = repeteability_factor * self.repeteability
        error = error_factor * self.error
        accuracy = error - repeteability
        new_value = (repeteability * (2 * np.random.random([duration, self.data.shape[-1]]) - 1) +
                    accuracy * (2 * np.random.random([duration, self.data.shape[-1]]) - 1)) * self.span
        for count, pos in enumerate(init_position):
            self.data[pos:pos+duration, count] +=  new_value[:,count]

    @PrepareData.step
    def add_frozen_data(self, duration=10):
        """
        Este método permite agregarle el comportamiento de instrumento congelado a la data
        duration: Cantidad de valores seguidos en la data en la que se mantenddrá la anomalía
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        for count, pos in enumerate(init_position):
            self.data[pos:pos+duration, count] = self.data[pos, count]

    @PrepareData.step
    def add_outliers(self, span_factor=0.03):
        """
        Este método permite agregarle outliers a la data

        min: Booleano que permite establecer si el outlier es positivo (False) o negativo (True)
        spanFactor: Fracción del span del instrumento que permitirá establecer la magnitud del outlier
        """
        init_position = np.random.randint(self.data.shape[0], size=(1, self.data.shape[-1]))[0]
        outlier = span_factor * self.span
        for count, pos in enumerate(init_position):
            min = bool(np.random.randint(2))
            if min:
                self.data[pos, count] -= outlier[count]
            else:
                self.data[pos, count] += outlier[count]

    @PrepareData.step
    def add_out_of_range(self, duration=10):
        """
        Este médoto permite agregarle la anomalía de instrumento fuera de rango a la data

        min: Es un booleano que indica si la anomalía se genera hacia el límite inferior (True) o superior (False).
        duration: Cantidad de valores seguidos en la data en la que se mantenddrá la anomalía
        """
        init_position = np.random.randint(self.data.shape[0] - duration, size=(1, self.data.shape[-1]))[0]
        anomaly = 0.07 * self.span
        for count, pos in enumerate(init_position):
            min = bool(np.random.randint(2))
            if min:
                self.data[pos:pos+duration, count] = self.lower_limit[count] - anomaly[count]
            else:
                self.data[pos:pos+duration, count] = self.upper_limit[count] + anomaly[count]

    def done(self, view=True, **options):
        """
        Este método permite graficar las variables que el usuario quiera visualizar
        """
        if view:
            self.view(columns=options['columns'], ylabel=options['ylabel'], xlabel=options['ylabel'])

    def __call__(self, decalibrations=0, sensor_drift=0, excesive_noise=0, frozen_data=0, outliers=0, out_of_range=0, add_WN=False, **options):
        """
        Este método permite generar las anomalías a la data para simular una data sintética proveniente de campo
        """

        "Agregando descalibración"
        for i in range(decalibrations):
            duration = np.random.randint(50, 500)
            self.add_decalibration(duration=duration)

        "Agregando sensor drift"
        for i in range(sensor_drift):
            duration = np.random.randint(50, 500)
            self.add_sensor_drift(duration=duration)

        "Agregando anomalía de ruido gaussiano"
        if add_WN:
            self.add_instrument_error()

        "Agregando anomalía de ruido gaussiano excesivo"
        for i in range(excesive_noise):
            duration = np.random.randint(50, 500)
            self.add_excesive_noise(duration=duration)

        "Agregando anomalía de instrumento congelado"
        for i in range(frozen_data):
            duration = np.random.randint(50, 500)
            self.add_frozen_data(duration=duration)

        "Agregando anomalía de outlier"
        for i in range(outliers):
            span_factor = np.random.randint(11) / 100
            self.add_outliers(span_factor=span_factor)

        "Agregando anomalía de instrumento fuera de rango"
        for i in range(out_of_range):
            duration = np.random.randint(50, 500)
            self.add_out_of_range(duration=duration)


        self.done(columns=[0, 1, 2, 3], ylabel='Amplitude', xlabel='Point')

        return self.data

    def round_by_dead_band(self, data):
        """
        Este método permite actualizar los valores medidos de acuerdo a la sensibilidad del instrumento especificado por
        la banda muerta de los instrumentos.
        deadband es la banda muerta de cada uno de los instrumentos, mientras que positions es una lista que indica cuál
        instrumento ha cambiado lo suficiente como para actualizar su valor.
        data: numpy.array mxn
        deadband: numpy.array 1xn
        positions: list
        """
        return np.array([np.round(data[:, count] * (10 **str(value)[::-1].find('.'))) / (10 **str(value)[::-1].find('.'))
                for count, value in enumerate(self.dead_band)])

    @typeCheckedAttribute.checkOptions
    def _check_options(self, **options):
        """
        Este método permite verificar las opciones incorporadas por el usuario por ejemplo:
        instrumentRange, instrumentError, instrumentDeadBand, instrumentRepeteability.
        la función define las opciones por defecto en caso que el usuario no las ingrese y al estar decorada con el decorador
        checkOptions permite agregarle funcionalidades.
        *args:
        **options:
        """
        if isinstance(self._data, pd.Series):
            sensors_numbers = 1
        else:
            sensors_numbers = self._data.shape[-1]
        " Definiendo los parámetros por defecto de los instrumentos"
        default_options = {key:np.zeros(sensors_numbers) for key in data_type_synthetic_data if key != '_data'}

        return default_options

    def _add_dead_band(self, *args):
        """
        Este método permite agregar el comportamiento de banda muerta del instrumento a la data, de tal manera que la data
        sintética cuente con el fenómeno de la sensibilidad del instrumento.

        data : numpy.array
        Ejemplo
                Sensor1  Sensor 2 Sensor3 ... Sensor n
        data = [[215.5,   321.5,   322,  ..., 225],          # tiempo 1
                [215.5,   321.5,   322,  ..., 225],          # tiempo 2
                [215.5,   321.5,   322,  ..., 225],          # tiempo 3
                [  . ,      .,      .,  ..., .  ],           # .
                [  . ,      .,      .,  ..., .  ],           # .
                [215.5,   321.5,   322,  ..., 225]]          # tiempo n
        """
        if len(args) == 0:
            data = self.data
        else:
            data = args[0]
        """
        Cálculo de la diferencia entre el valor siguiente y el anterior de cada instrumento para verificar la maginitud
        de cuánto ha cambiado la variable
        """
        difference = np.diff(data, axis=0)
        "Posiciones donde el nuevo valor ha superado la sensibilidad del instrumento"
        pos_true = abs(difference) >= self.dead_band
        "Posiciones donde el nuevo valor no ha cambiado lo suficiente como para que el instrumento lo capte"
        pos_false = abs(difference) < self.dead_band
        "Numero de decimales a los que se redondearán cada valor de los instrumentos"
        decimals = [str(self.dead_band[count])[::-1].find('.') for count in range(self.dead_band.shape[-1])]
        "Aplicación del redondeo de las variables y la sensibilidad del instrumento"
        for count in range(pos_true.shape[-1]):
            data[1::,:][pos_false[:,count], count] = np.round(data[:-1:,:][pos_false[:,count], count], decimals[count])
            data[1::,:][pos_true[:,count],count] = np.round(data[1::,:][pos_true[:,count],count], decimals[count])

        if len(args) == 0:
            self.data = data
        else:
            return data

    def view(self, columns=[0], xlabel='Time', ylabel='Amplitude'):
        """

        """
        plt.plot(self.data[:, columns])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show(block=True)