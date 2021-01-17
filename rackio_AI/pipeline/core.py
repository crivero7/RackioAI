from easy_deco.core import decorator
import inspect
from rackio_AI.core import RackioAI


class StopPipeline(Exception):
    pass


class Pipeline(object):
    """
    Chain stages together. Assumes the last is the consumer.
    """

    app = RackioAI()

    def __init__(self):
        """
        Documentation here
        """
        self._pipeline = None

    def __call__(self, func_args, *args):
        """
        Documentation here
        """
        # Class definitions
        f = self.sink(args[-1])
        _consumer = Func(f, *func_args[-1]["args"], **func_args[-1]["kwargs"])

        c = self.consumer(_consumer)
        c.__next__() 
        t = c

        filter_args = list(reversed(func_args[1:-1]))
        for i, stg in enumerate(reversed(args[1:-1])):
            f = self.del_attr(stg)
            _filter = Func(f, *filter_args[i]["args"], **filter_args[i]["kwargs"])
            s = self.stage(_filter, t)
            s.__next__() 
            t = s

        f = self.del_attr(args[0])
        _producer = Func(f, *func_args[0]["args"], **func_args[0]["kwargs"])
        p = self.producer(_producer, t)
        p.__next__() 
        self._pipeline = p
        return self._pipeline

    def start(self, initial_state):
        try:
            self._pipeline.send(initial_state)
        except StopPipeline:
            self._pipeline.close()

    @staticmethod
    def producer(f, n):
        """
        Producer: only .send (and yield as entry point)
        :param f:
        :param n:
        :return:
        """

        state = (yield)  # get initial state
        while True:
            try:
                res = f(state)
            except StopPipeline:
                return

            n.send(res)

    @staticmethod
    def stage(f, n):
        """
        Stage: both (yield) and .send.
        :param f:
        :param n:
        :return:
        """

        while True:
            r = (yield)
            n.send(f(r))

    @staticmethod
    def consumer(f):
        """
        Consumer: only (yield).
        :param f:
        :return:
        """

        while True:
            r = (yield)
            data = f(r)
            data.info()
            self.app._data = data

    @staticmethod
    def sink(f):
        """
        Documentation here
        """
    
        def wrapper(*args, **kwargs):

            f(*args, **kwargs)

            raise StopPipeline('Enough!')

        return wrapper
    
    def del_attr(self, f):
        """
        Documentation here
        """
    
        def wrapper(*args, **kwargs):

            result = f(*args, **kwargs)

            attrs = inspect.getmembers(self, lambda variable:not(inspect.isroutine(variable)))

            for attr, value in attrs:

                if attr.startswith('_') and attr.endswith('_'):
                    
                    if not(attr.startswith('__')) and not(attr.endswith('__')):

                        delattr(self, attr)

            self.data = result

            return result

        return wrapper

class Func(object):
    """
    Documentation here
    """

    def __init__(self, f, *args, **kwargs):
        """
        Documentation here
        """
        self._function = f
        self._args = args
        self._kwargs = kwargs

    def __call__(self, data):
        """
        Documentation here
        """

        return self._function(data, *self._args, **self._kwargs)
