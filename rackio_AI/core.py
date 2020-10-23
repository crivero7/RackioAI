import os, pickle
import pandas as pd
import numpy as np
from rackio_AI._singleton import Singleton
from rackio_AI.managers.preprocess import PreprocessManager
from rackio_AI.rackio_loader.rackio_tpl import TPL
from rackio_AI.data_handler import DataHandler
from rackio_AI.preprocessing.synthetic_data import SyntheticData


class RackioAI(Singleton):
    """
    This is the main class of the package.

    **RackioAI** is an open source, MIT License [Rackio-Framework](https://github.com/rack-io/rackio-framework) extension
    to do data analysis (based on [Pandas](https://pandas.pydata.org/)) and deep learning models (based on [Keras](https://keras.io/))
    taking advantage of **Rackio** system architecture.

    You can use it by the following snippet code:

    >>> from rackio import Rackio
    >>> from rackio_AI import RackioAI
    >>> app = Rackio()
    >>> RackioAI(app)
    """

    def __init__(self):
        super(RackioAI, self).__init__()
        self.loader = TPL()
        self.synthetic_data = SyntheticData()
        self.data_handler = DataHandler(self._observer)
        self._preprocess_manager = PreprocessManager()
        self.app = None

    def _observer(self, value):
        """

        """
        self._data = value

    def __call__(self, app):
        """

        """
        self.app = app

    def load(self, filename):
        """
        filename (str):
        """
        if os.path.isdir(filename) or os.path.isfile(filename):

            if filename.endswith('.pkl'):
                try:
                    with open(filename, 'rb') as file:
                        data = pickle.load(file)
                except:
                    try:
                        with open(filename, 'rb') as file:
                            data = pd.read_pickle(file)
                    except:
                        raise ImportError('{} is not possible loaded it'.format(filename))

                self.data = data
                return data

            try:
                self._load_data(filename)
                data = self.loader.to('dataframe')
                self.data = data
                return data

            except:
                raise ImportError('{} is not possible loaded because is no a .tpl file'.format(filename))

        else:
            raise TypeError('You can only load .tpl or .pkl files')

    @property
    def data(self):
        """

        """
        return self._data

    @data.setter
    def data(self, value):
        """
        value (pd.DataFrame or np.ndarray):
        """
        if isinstance(value, pd.DataFrame) or isinstance(value, np.ndarray):

            if isinstance(value, np.ndarray):
                value = pd.DataFrame(value)

            self.synthetic_data.data = value
            self.data_handler.data = value
        else:
            raise TypeError('value must be a pd.DataFrame or np.ndarray')

    def append_preprocess_model(self, preprocess_model):
        """Append a preprocessing model to the preprocessing manager.

        # Parameters
        preprocessing_model (Preprocess): a Preprocess object.
        """

        self._preprocess_manager.append_preprocessing(preprocess_model)

    def get_preprocess(self, name):
        """Returns a RackioAI preprocess model defined by its name.

        # Parameters
        name (str): a RackioAI preprocess name.
        """

        return self._preprocess_manager.get_preprocessing_model(name)

    def serialize_preprocess(self, name):
        """

        """
        preprocess = self.get_preprocess(name)

        return preprocess.serialize()

    def summary(self):
        """
        Returns a RackioAI Application Summary (dict).
        """
        result = dict()
        result["preprocessing_manager"] = self._preprocess_manager.summary()

        return result

    @staticmethod
    def save_obj(obj, filename, format='pkl'):
        """
        Method to persist any object
        params:
            obj: (obj) any object persistable
            filename: (str) with no extension
            format: (str) with no dot (.) at the beginning
        """
        if format.lower()=='pkl':
            with open('{}.{}'.format(filename,format), 'wb') as file:
                pickle.dump(obj, file)

    @staticmethod
    def load_obj(filename, format='pkl'):
        """
        Method to load any saved object with RackioAI's save method
        params:
            filename: (str) with no extension
            format: (str) with no dot (.) at the beginning

        return:
            obj: (obj)
        """
        obj = None
        if format.lower()=='pkl':
            with open('{}.{}'.format(filename,format), 'rb') as file:
                obj = pickle.load(file)

        return obj

    def test_data(self, name='Leak'):
        """

        """
        os.chdir('..')
        cwd = os.getcwd()
        filename = os.path.join(cwd, 'rackio_AI', 'data', name)
        self._load_data(filename)
        return self.loader.to('dataframe')

    def _load_data(self, filename):
        """

        """
        return self.loader.read(filename)

if __name__=="__main__":
    import doctest
    doctest.testmod()