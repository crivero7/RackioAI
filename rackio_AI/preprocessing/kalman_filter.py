class KalmanFilter:
    """
    Class to filter data using kalman filter

    **Attributes**

    * **alpha:** (float) (default=1.0)
    * **beta:** (float) (default=0.0)
    * **f_value:** (float)
    """

    def __init__(self):
        self.alpha = 1.0
        self.beta = 0.0
        self.filtered_value = 0.0
        self.posteri_error_estimate = 0.0

    def set_init_value(self, value):
        """
        set init value for the Kalman filter

        **Parameters**

        * **value:** (float)

        **return**

            None

        ```python
        >>> import numpy as np
        >>> from rackio_AI.preprocessing.core import Preprocessing
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Kalman Filter', description='test for filter data', problem_type='regression')
        >>> kf = preprocess.kalman_filter # Kalman filter definition
        >>> variable_to_filter = np.ones((10,1)) +np.random.random((10,1))
        >>> kf.set_init_value(variable_to_filter[0])

        ```
        """
        self.filtered_value = value

    def __call__(self, value):
        """

        **Parameters**

        * **value:** (float) value to filter

        **return**

        * **value: **(float) filtered value

        See [This example](https://github.com/crivero7/RackioAI/blob/main/examples/example9.py) for a real example

        ```python
        >>> import numpy as np
        >>> from rackio_AI.preprocessing.core import Preprocessing
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Kalman Filter', description='test for filter data', problem_type='regression')
        >>> kf = preprocess.kalman_filter # Kalman filter definition
        >>> kf.alpha = 0.001
        >>> kf.beta = 0.2
        >>> variable_to_filter = np.ones((10,1)) +np.random.random((10,1))
        >>> filtered_variable = np.array([kf(value) for value in variable_to_filter]) # Applying Kalman filter

        ```
        """

        f_value = self.filtered_value
        priori_error_estimate = self.posteri_error_estimate + self.alpha

        blending_factor = priori_error_estimate / (priori_error_estimate + self.beta)
        self.filtered_value = f_value + blending_factor * (value - f_value)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.filtered_value

if __name__=="__main__":
    import doctest
    doctest.testmod()