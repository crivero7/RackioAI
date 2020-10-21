from rackio_AI.decorators import typeCheckedAttribute
import numpy as np
import pandas as pd
import functools
from abc import abstractmethod, ABCMeta


@typeCheckedAttribute.typeassert(data=[pd.Series, pd.DataFrame, np.ndarray])
class PrepareData(metaclass=ABCMeta):

    def __init__(self):
        pass

    def __str__(self):
        return '{}'.format(self.__dict__)

    @staticmethod
    def step(function=None, *args, **kwargs):
        def decorator(function):
            @functools.wraps(function)
            def wrap(*args, **options):

                function(*args, **options)

            return wrap

        if function is None:
            return decorator
        else:
            return decorator(function)

    @abstractmethod
    def done(self, *args, **kwargs):
        pass