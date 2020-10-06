class KalmanFilter(object):

    def __init__(self):
        self._process_variance = 1.0
        self._estimated_measurement_variance = 0.1
        self._posteri_estimate = 0.0
        self._posteri_error_estimate = 1.0

    @property
    def process_variance(self):
        """

        """
        return self._process_variance

    @process_variance.setter
    def process_variance(self, value):
        """

        """
        self._process_variance = value

    @property
    def estimated_measurement_variance(self):
        """

        """
        return self._estimated_measurement_variance

    @estimated_measurement_variance.setter
    def estimated_measurement_variance(self, value):
        """

        """
        self._estimated_measurement_variance = value

    def __call__(self, value):
        """

        """
        priori_estimate = self._posteri_estimate
        priori_error_estimate = self._posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self._posteri_estimate = priori_estimate + blending_factor * (value - priori_estimate)
        self._posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self._posteri_estimate