from .scaler import Scaler
from .kalman_filter import KalmanFilter
from rackio_AI.core import RackioAI


class Preprocessing:

    app = RackioAI()

    def __init__(self, name, description, problem_type='regression'):
        """
        name (str): preprocessing model's name

        """
        self._data = None
        self._name = name
        self._description = description
        self._type = problem_type

        if problem_type.lower() in ['regression', 'classification']:

            if problem_type.lower() == 'regression':
                self.preprocess = Regression(name, description)

            else:

                self.preprocess = Classification(name, description)

        self.scaler = Scaler()
        self.kalman_filter = KalmanFilter()

    def __call__(self, action, data):
        """

        """
        allowed_actions = ["scaler", "prepare", "split"]

        if action.lower() in allowed_actions:

            todo = getattr(self.preprocess,action.lower())

            todo(data)
        else:
            raise NotImplementedError("{} method is not implemented in {} class".format(action, self._preprocesssing.get_type().get_name()))

    def serialize(self):
        """

        """
        result = {"name": self.get_name(),
                  "description": self.description,
                  "type": self._type}

        return result

    @property
    def data(self):
        return self.app._data

    @data.setter
    def data(self, value):
        self.app._data = value

    def get_name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    def get_name(self):

        return self._name


class Regression(Preprocessing):

    def __init__(self, name, description):
        """



        """
        #super(Regression, self).__init__(name, description)


class Classification(Preprocessing):

    def __init__(self, name, description):
        """

        """
        #super(Classification, self).__init__(name, description)