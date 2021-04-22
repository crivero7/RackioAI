"The Factory Concept"

from abc import ABCMeta, abstractmethod
from .regression import RackioRegression
import pandas as pd
import numpy as np
import tensorflow as tf


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
    def predict():
        r"""
        Documentation here
        """
        pass

    @abstractmethod
    def evaluate():
        r"""
        Documentation here
        """
        pass

    @abstractmethod
    def retrain():
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


class RackioDNN:
    r"""
    The Factory Class
    """

    @staticmethod
    def create(
        model:str, 
        units:list, 
        activations:list, 
        scaler=None, 
        **kwargs
        ):
        r"""
        A static method to get a concrete RackioLSTM model
        """
        if model.lower() == 'regression':
            return RackioRegression(
                units, 
                activations, 
                scaler=scaler, 
                **kwargs
            )

    def predict(self, x):
        r"""
        Documentation here
        """
        pass

    def evaluate(self, x, y, **kwargs):
        r"""
        Documentation here
        """
        pass

    def retrain(self, x, y, **kwargs):
        r"""
        Documentation here
        """
        pass

    def load(self, directory, **kwargs):
        r"""
        Documentation here
        """
        self._model = tf.keras.models.load_model(directory, **kwargs)

        return self._model
