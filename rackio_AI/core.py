import os, pickle
import pandas as pd
from ._singleton import Singleton
from .managers.preprocess import PreprocessManager
from .rackio_loader.rackio_tpl import TPL
from .preprocessing.synthetic_data import SyntheticData


class RackioAI(Singleton):

    def __init__(self):
        """

        """
        super(RackioAI, self).__init__()

        self.loader = TPL()
        self.synthetic_data = SyntheticData()
        self._preprocess_manager = PreprocessManager()

        self.app = None

    def __call__(self, app):
        """

        """
        self.app = app

    def load_data(self, filename):
        """

        """
        if not hasattr(self, 'convert_data_to'):
            setattr(self, 'convert_data_to', self.loader.to)

        return self.loader.read(filename)

    @property
    def data(self):
        """

        """
        return self._data

    @data.setter
    def data(self, filename):
        """
        filename (str):
        """
        if os.path.isdir(filename):
            self._data = self.load_data(filename)

        elif os.path.isfile(filename):

            if filename.endswith('.tpl'):
                data = self.load_data(filename)

                self._data = data

            elif filename.endswith('.pkl'):
                try:
                    with open(filename, 'rb') as file:
                        data = pickle.load(file)
                        self.synthetic_data.data = data
                        self._data = data
                except:
                    try:
                        with open(filename, 'rb') as file:
                            data = pd.read_pickle(file)
                            self.synthetic_data.data = data
                            self._data = data
                    except:
                        raise ImportError('{} is not possible loaded it'.format(filename))


    def set_data(self, filename):
        """
        filename (str):
        """

        self._data = self.load_data(filename)

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
