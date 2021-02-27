class KalmanFilter:
    """
    Class to filter data using kalman filter

    **Attributes**

    * **alpha:** (float) (default=1.0)
    * **beta:** (float) (default=0.0): Uncertainty in measurement
    * **f_value:** (float)
    """

    def __init__(self):
        """

        """
        self.alpha = 1.0
        self.beta = 0.0
        self.filtered_value = 0.0
        self.posteri_error_estimate = 0.0

    def set_init_value(self, value):
        """
        Set init value for the Kalman filter

        ___
        **Parameters**

        * **:param value:** (float)

        * **:return:**

        None

        ___

        ## Snippet code

        ```python
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> preprocessing = RackioAI.get('Preprocessing', _type="Preprocessing")
        >>> kf = preprocessing.kalman_filter # Kalman filter definition
        >>> variable_to_filter = np.ones((10,1)) + np.random.random((10,1))
        >>> kf.set_init_value(variable_to_filter[0])

        ```
        """
        self.filtered_value = value

    def __call__(self, value):
        """
        **Parameters**

        * **:param value:** (float) value to filter
        :return:

        See [This example](https://github.com/crivero7/RackioAI/blob/main/examples/example9.py) for a real example

        ```python
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> preprocessing = RackioAI.get("Preprocessing", _type="Preprocessing")
        >>> kf = preprocessing.kalman_filter # Kalman filter definition
        >>> kf.alpha = 0.001
        >>> kf.beta = 0.2
        >>> variable_to_filter = np.ones((10,1)) + np.random.random((10,1))
        >>> filtered_variable = np.array([kf(value) for value in variable_to_filter]) # Applying Kalman filter

        ```
        """
        # Prediction
        f_value = self.filtered_value
        priori_error_estimate = self.posteri_error_estimate + self.alpha

        # Correction
        blending_factor = priori_error_estimate / (priori_error_estimate + self.beta)
        self.filtered_value = f_value + blending_factor * (value - f_value)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.filtered_value
