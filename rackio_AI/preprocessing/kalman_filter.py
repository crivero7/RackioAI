class KalmanFilter(object):
    """
    alpha = 1.0 (default)
    beta = 0.0 (default)
    """

    def __init__(self):
        self._alpha = 1.0
        self._beta = 0.0
        self._init_value = 0.0
        self._posteri_error_estimate = 0.0

    @property
    def alpha(self):
        """

        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """

        """
        self._alpha = value

    @property
    def beta(self):
        """

        """
        return self._beta

    @beta.setter
    def beta(self, value):
        """

        """
        self._beta = value

    @property
    def init_value(self):
        """

        """
        return self._init_value

    @init_value.setter
    def init_value(self, value):
        """

        """
        self._init_value = value

    def __call__(self, value):
        """

        """
        init_value = self._init_value
        priori_error_estimate = self._posteri_error_estimate + self.alpha

        blending_factor = priori_error_estimate / (priori_error_estimate + self.beta)
        self._init_value = init_value + blending_factor * (value - init_value)
        self._posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self._init_value