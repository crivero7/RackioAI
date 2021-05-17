"The Factory Concept"

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from .lstm import RackioLSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .classification import RackioClassification


class FactoryRackioDNN(metaclass=ABCMeta):
    """
    RackioLSTM Interface class
    """
    @staticmethod
    @abstractmethod
    def create():
        r"""
        Documentation here
        """
        pass

    @abstractmethod
    def load():
        r"""
        Documentation here
        """
        pass


class RackioDNN(FactoryRackioDNN):
    r"""
    The Factory Class
    """
    def __init__(self):
        r"""
        Documentation here
        """
        self._model = None

    @staticmethod
    def create(
        model:str, 
        units:list, 
        activations:list, 
        min_max_values=None, 
        **kwargs
        ):
        r"""
        A static method to get a concrete RackioLSTM model
        """
        if model.lower() == 'lstm':
            
            return RackioLSTM(
                units, 
                activations, 
                min_max_values=min_max_values, 
                **kwargs
            )

        if model.lower() == 'classification':

            return RackioClassification(
                units, 
                activations, 
                **kwargs
            )

    @classmethod
    def load(cls, directory, **kwargs):
        r"""
        Documentation here
        """
        cls._model = tf.keras.models.load_model(directory, **kwargs)
        return cls._model

    @classmethod
    def predict(cls, X):
        r"""
        Documentation here
        """
        return cls._model.predict(X)

    @classmethod
    def plot(cls, dataset, dataset_type='testing'):
        r"""
        Documentation here
        """
        X, y = dataset['train_dataset']
        
        if dataset_type.lower()=="testing":
            
            X, y = dataset['test_dataset']
        
        y = y.reshape((y.shape[0], 1))
        y_predict = cls.predict(X)
        _result = np.concatenate((y_predict, y), axis=1)
        result = pd.DataFrame(_result, columns=['Prediction', 'Original'])
        result.plot(kind='line')
        plt.show()