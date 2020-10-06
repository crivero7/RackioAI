from .scaler import Scaler
from .kalman_filter import KalmanFilter


class Preprocessing:

    def __init__(self, name, description):
        """

        """
        self._data = None
        self._name = name
        self._description = description

        self.scaler = Scaler()
        self.kalman_filter = KalmanFilter()