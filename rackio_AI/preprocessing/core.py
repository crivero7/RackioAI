from .scaler import Scaler


class Preprocessing:

    def __init__(self, name, description):
        """

        """
        self._data = None
        self._name = name
        self._description = description

        self.scaler = Scaler()