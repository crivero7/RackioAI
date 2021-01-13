"""
Pipeline
(yield) -> receiver
.send -> producer
Provide initial state to producer, avoiding globals.
Stop iteration after a bit.
Wrap in nice class.
"""


class StopPipeline(Exception):
    pass


class Pipeline(object):
    """
    Chain stages together. Assumes the last is the consumer.
    """

    def __init__(self, *args, **kwargs):
        """
        Documentation here
        """
        class_args = kwargs["args"]
        # Class definitions
        _consumer = Func(args[-1], *class_args[-1]["args"], **class_args[-1]["kwargs"])

        c = Pipeline.consumer(_consumer)
        c.__next__() 
        t = c

        filter_args = list(reversed(class_args[1:-1]))
        for i, stg in enumerate(reversed(args[1:-1])):
            _filter = Func(stg, *filter_args[i]["args"], **filter_args[i]["kwargs"])
            s = Pipeline.stage(_filter, t)
            s.__next__() 
            t = s

        _producer = Func(args[0], *class_args[0]["args"], **class_args[0]["kwargs"])
        p = Pipeline.producer(_producer, t)
        p.__next__() 
        self._pipeline = p

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
            f(r)


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