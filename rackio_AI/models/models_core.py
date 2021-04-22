"The Factory Concept"

from abc import ABCMeta, abstractmethod
from .acunet import AcuNet
import tensorflow as tf

class FactoryRackioLSTM(metaclass=ABCMeta):
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


class RackioLSTM:
    r"""
    The Factory Class
    """
    @staticmethod
    def create(
        model:str, 
        units:list, 
        activations:list, 
        compile_options={
            'optimizer': tf.keras.optimizers.Adam(
                learning_rate=0.1, 
                amsgrad=True
            ),
            'loss': 'mse',
            'metrics': tf.keras.metrics.MeanAbsoluteError()
        }, 
        scaler=None, 
        **kwargs
        ):
        r"""
        A static method to get a concrete RackioLSTM model
        """
        if model.lower() == 'acunet':
            return AcuNet(
                units, 
                activations, 
                compile_options=compile_options, 
                scaler=scaler, 
                **kwargs
            )
        
        elif model.lower() == 'pfm':
            return PFMNet(
                units, 
                activations, 
                compile_options=compile_options, 
                scaler=scaler, 
                **kwargs
            )

        elif model.lower() == 'sa':
            return SANet(
                units, 
                activations, 
                compile_options=compile_options, 
                scaler=scaler, 
                **kwargs
            )

        elif model.lower() == 'claus tail gas':
            return ClausTGNet(
                units, 
                activations, 
                compile_options=compile_options, 
                scaler=scaler, 
                **kwargs
            )
        
        elif model.lower() == 'observer':
            return ObserverNet(
                units, 
                activations, 
                compile_options=compile_options, 
                scaler=scaler, 
                **kwargs
            )
