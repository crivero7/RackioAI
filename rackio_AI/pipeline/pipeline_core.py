from easy_deco.core import decorator
import inspect
from rackio_AI.core import RackioAI


class StopPipeline(Exception):
    """
    Documentation here
    """
    pass


class Pipeline(object):
    """
    Pipeline is a class based on a simple architectural style called Pipe and Filter, which connects a number
    of components that process a stream of data, each connected to the next component in the processing pipeline
    via a Pipe.

    The Pipe and Filter architecture is inspired by the Unix technique of connecting  the output of an application
    to the input of another via pipes on the shell.

    The Pipe and Filter architecture consists of one or more data sources. The data source is connected to data
    filters via pipes. Filters process the data they receive, passing them to others filters in the pipeline. The
    final data is received at a Data Sink.

    In the following image you can see graphically this architecture style.

    ![Pipe and Filter Architecture](../img/PipeAndFilter.png)

    Pipe and filter are used commonly for applications that perform a lot of data processing such as data analytics,
    data transformation, metadata extraction, and so on.
    """

    app = RackioAI()

    def __call__(self, func_args, *args):
        """
        Pipeline is too a callable object, so, when the Pipeline is called this function creates the pipeline
        architecture. Each component in the pipeline (function or method) may need input arguments in addition to its
        main argument (data to be processed), so, this arguments differents to the main argument must be passed into
        pipeline as a list of dicts with the following structure.

        ```python
        func_args = [
            {
                "args": [],
                "kwargs": {},
            },
            {
                "args": [],
                "kwargs": {},
            },
            {
                "args": [],
                "kwargs": {},
            }
        ]
        ```

        Each element in func_args represents the input arguments of each component (function or method) in the 
        pipeline.

        Each component (function or method) are passed into pipeline after the first argument (func_args) in the
        callable. So, the first element in *func_args* represents the arguments for the first component of the 
        pipeline

        ___

        **Parameters**

        * **func_args:** (list of dicts) Functions argument of each component in the pipeline
        * **args:** (function or method) Positional arguments that represents each component in the pipeline.

        **returns**

        * **obj:** The main argument passed into each component.

        ## Snippet code

        ```python
        >>> from rackio_AI import Pipeline
        >>> import numpy as np

        >>> def load(value): return np.array([value, value, value, value])

        >>> def power(data, value): return data ** value

        >>> def sum(data, value): return data + value

        >>> args = [{"args": [], "kwargs": {}}, {"args": [2], "kwargs": {}}, {"args": [1], "kwargs": {}}, {"args": [-2], "kwargs": {}}]
        >>> pipeline = Pipeline()
        >>> pipeline(args, load, power, sum, sum)
        >>> pipeline.start(2)
        >>> pipeline.data
        array([3, 3, 3, 3], dtype=int32)

        ```
        """
        # Define source component
        f = self._sink(args[-1])
        _sink = Func(f, *func_args[-1]["args"], **func_args[-1]["kwargs"])
        c = self.__sink(_sink)
        c.__next__() 
        t = c

        # Define filters component
        filter_args = list(reversed(func_args[1:-1]))
        for i, stg in enumerate(reversed(args[1:-1])):
            _filter = Func(stg, *filter_args[i]["args"], **filter_args[i]["kwargs"])
            s = self.__filter(_filter, t)
            s.__next__() 
            t = s

        # Define source component
        _source = Func(args[0], *func_args[0]["args"], **func_args[0]["kwargs"])
        p = self.__source(_source, t)
        p.__next__() 
        self._pipeline = p

        return 

    def start(self, initial_state):
        """
        This method starts to run the pipeline architecture

        **Parameters**

        * **initial_state:** First argument of the pipeline's source method

        **returns**

        None

        """
        try:
            
            self._pipeline.send(initial_state)
        
        except StopPipeline:
            
            self._pipeline.close()

    @staticmethod
    def __source(f, n):
        """
        It's the first component of the pipeline

        **Parameters**

        * **:param f:** (Function or method)
        * **:param n:** (generator) Function generate for the following component

        **returns**
        
        None
        """
        state = (yield)  # get initial state
        
        while True:
            
            try:
                
                res = f(state)
            
            except StopPipeline:
                
                return

            n.send(res)

    @staticmethod
    def __filter(f, n):
        """
        Filter component of the pipeline, this component receive a stream data from the previous component, 
        it processes it and it sends to the next component.

        **Parameters**

        * **:param f:** (Funtion or method)
        * **:param n:** (generator) Function generate for the following component
        
        **returns**
        
        None
        """
        while True:
            
            r = (yield)
            n.send(f(r))

    @staticmethod
    def __sink(f):
        """
        Sink of the pipeline, this component only receive a stream data from the previous component, 
        it processes it and it finishes the pipeline.

        **Parameters**

        * **:param f:** (Funtion or method)
        
        **returns**
        
        None
        """
        while True:
            
            r = (yield)
            f(r)

    def _sink(self, f):
        """
        Sink decorator to stop the pipeline execution

        **Parameters**

        * **:param f:** (Function or method) at the end of the pipeline

        * **returns**

        * **wrapper:** (Decorated function)
        """

        def wrapper(*args, **kwargs):

            data = f(*args, **kwargs)

            self.data = data

            raise StopPipeline('Enough!')

        return wrapper
    
    @property
    def data(self):
        """
        This attribute storage the stream data processed in the pipeline
        """
        return self.app._data

    @data.setter
    def data(self, value):
        """
        This attribute storage the stream data processed in the pipeline

        Setter property method to send the stream data in the pipeline to RackioAI app

        **Parameters**

        * **:param value:** (obj) stream data processed in the pipeline
        """
        self.app._data = value


class Func(object):
    """
    Class decorator to allow pass arguments to each component in te pipeline

    ```python
    Func(f, *args, **kwargs)
    ```

    **Parameters**

    * **:param f:** (Function or method) Component of the pipeline
    * **:param args:** Positional arguments, except the first argument of the component *f* in the pipeline 
    * **:param kwargs** Keyword arguments for the component *f* in the pipeline
    """

    def __init__(self, f, *args, **kwargs):
        self._function = f
        self._args = args
        self._kwargs = kwargs

    def __call__(self, data):
        """
        Callable to execute the processing of each pipeline component

        **Parameters**

        * **:param data:** Data stream through the pipeline

        **returns**

        * **data**: Data stream processed in each pipeline component
        """

        return self._function(data, *self._args, **self._kwargs)

if __name__ == "__main__":
    import doctest

    doctest.testmod()