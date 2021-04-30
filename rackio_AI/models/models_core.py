"The Factory Concept"

from abc import ABCMeta, abstractmethod
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
    def load():
        r"""
        Documentation here
        """
        pass


class RackioDNN(FactoryRackioDNN):
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
        if model.lower() == 'lstm':
            from .lstm import RackioLSTM
            return RackioLSTM(
                units, 
                activations, 
                scaler=scaler, 
                **kwargs
            )

    def load(self, directory, **kwargs):
        r"""
        Documentation here
        """
        return tf.keras.models.load_model(directory, **kwargs)