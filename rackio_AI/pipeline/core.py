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

    def __init__(self, *args):
        c = Pipeline.consumer(args[-1])
        c.__next__() 
        t = c
        for stg in reversed(args[1:-1]):
            s = Pipeline.stage(stg, t)
            s.__next__() 
            t = s
        p = Pipeline.producer(args[0], t)
        p.__next__() 
        self._pipeline = p

    def start(self, initial_state):
        try:
            self._pipeline.send(initial_state)
        except StopIteration:
            self._pipeline.close()

    @staticmethod
    def producer(f, n, *args, **kwargs):
        """
        Producer: only .send (and yield as entry point)
        :param f:
        :param n:
        :return:
        """

        state = (yield)  # get initial state
        while True:
            try:
                res, state = f(state, *args, **kwargs)
            except StopPipeline:
                return

            n.send(res, *args, **kwargs)

    @staticmethod
    def stage(f, n, *args, **kwargs):
        """
        Stage: both (yield) and .send.
        :param f:
        :param n:
        :return:
        """

        while True:
            r = (yield)
            n.send(f(r, *args, **kwargs))

    @staticmethod
    def consumer(f, *args, **kwargs):
        """
        Consumer: only (yield).
        :param f:
        :return:
        """

        while True:
            r = (yield)
            f(r, *args, **kwargs)


class Filter(object):
    """
    Documentation here
    """

    def __init__(self, f, n, *args, **kwargs):
        """
        Documentation here
        """
        self._function = f
        self._next_function = n
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
