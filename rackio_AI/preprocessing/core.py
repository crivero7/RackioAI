from rackio_AI.core import RackioAI
from .scaler import Scaler
from .kalman_filter import KalmanFilter


class Preprocessing:
    """
    Description here
    """

    app = RackioAI()

    def __init__(self, name, description, problem_type='regression'):
        """
        ...Description here...

        **Parameters**

        * **:param name:**
        * **:param description:**
        * **:param problem_type:**
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

    @property
    def data(self):
        """
        ...Description here...

        **:return:**

        """
        return self.app._data

    @data.setter
    def data(self, value):
        """

        :param value:
        :return:
        """
        self.app._data = value

    @property
    def description(self):
        """
        ...Description here...

        **Parameters**

        None

        **:return:**

        """
        return self._description

    @description.setter
    def description(self, value):
        """

        :param value:
        :return:
        """
        self._description = value

    def __call__(self, action, data):
        """
        ...Description here...

        **Parameters**

        * **:param action:**
        * **:param data:**

        **:return:**

        """
        allowed_actions = ["scaler", "prepare", "split"]

        if action.lower() in allowed_actions:

            todo = getattr(self.preprocess,action.lower())

            todo(data)
        else:
            raise NotImplementedError("{} method is not implemented in {} class".format(action, self._preprocesssing.get_type().get_name()))

    def serialize(self):
        """
        ...Description here..

        **Parameters**

        None

        **:return:**


        """
        result = {"name": self.get_name(),
                  "description": self.description,
                  "type": self._type}

        return result

    def get_name(self):
        """
        ...Description here...

        **Parameters**

        None

        **:return:**

        * **name:** (str) Preprocessing name
        """
        return self._name


class Regression(Preprocessing):
    """
    ...Description here...

    """

    def __init__(self, name, description):
        """
        ...Description here...

        **Parameters**

        * **:param name:**
        * **:param description:**
        """
        self._name = name
        self._description = description


class Classification(Preprocessing):
    """
    ...Description here...

    """

    def __init__(self, name, description):
        """
        ...Description here...

        **Parameters**

        * **:param name:**
        * **:param description:**
        """
        self._name = name
        self._description = description