"The Factory Concept"

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from .lstm import RackioLSTM
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from .classification import RackioClassification
from .observer import RackioObserver


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
    def __init__(self, mode='training'):
        r"""
        Documentation here
        """
        self._model = None
        self._models = dict()
        if mode in ['training', 'production']:
            self._mode = mode

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

        if model.lower() == 'observer':

            return RackioObserver(
                units, 
                activations,
                min_max_values=min_max_values, 
                **kwargs
            )

    @classmethod
    def load(cls, directory, **kwargs):
        r"""
        Documentation here
        """
            
        cls._model = tf.keras.models.load_model(directory, **kwargs)
        
        return cls._model

    def models_load(self, root, **kwargs):
        r"""
        Documentation here
        """
        directories = [ name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)) ]

        for directory in directories:
            
            dir = os.path.join(root, directory)
            filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
            
            if any(file.endswith(".pb") for file in filenames):
                
                self._models[directory] = RackioDNN.load(dir)

        return self._models

    @classmethod
    def predict(cls, X):
        r"""
        Documentation here
        """
        return cls._model.predict(X)

    def run(self, X, active_model):
        r"""
        Documentation here
        """

        Y = self._models[active_model].predict(X)

        return Y

    @classmethod
    def plot(cls, dataset, dataset_type='testing', plotting_backend='matplotlib'):
        r"""
        Documentation here
        """
        X, y = dataset['train_dataset']
        
        if dataset_type.lower()=="testing":
            
            X, y = dataset['test_dataset']

        if dataset_type.lower()=="all":
            X_test, y_test = dataset['test_dataset']
            X , y = np.concatenate((X, X_test), axis=0), np.concatenate((y, y_test), axis=0)
        
        y = y.reshape((y.shape[0], 1))
        y_predict = cls.predict(X)
        _result = np.concatenate((y_predict, y), axis=1)
        result = pd.DataFrame(_result, columns=['Prediction', 'Original'])

        if plotting_backend=="matplotlib":
            result.plot(kind='line')
            plt.show()
        
        if plotting_backend=="plotly":
            pd.options.plotting.backend = "plotly"
            fig = result.plot(kind='line')
            fig.show()

        return result

    @classmethod
    def get_predict(cls, dataset, dataset_type='all'):
        r"""
        Documentation here
        """
        X, y = dataset['train_dataset']
        
        if dataset_type.lower()=="testing":
            
            X, y = dataset['test_dataset']

        if dataset_type.lower()=="all":
            X_test, y_test = dataset['test_dataset']
            X , y = np.concatenate((X, X_test), axis=0), np.concatenate((y, y_test), axis=0)
        
        y = y.reshape((y.shape[0], 1))
        y_predict = cls.predict(X)
        _result = np.concatenate((y_predict, y), axis=1)
        return pd.DataFrame(_result, columns=['Prediction', 'Original'])