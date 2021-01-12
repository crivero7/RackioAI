"""Pipeline example
"""
from rackio_AI import Pipeline, StopPipeline

def produce(state):
    """
    Given a state, produce a result and the next state.
    :param state:
    :return:
    """
    import time
    if state == 3:
        raise StopPipeline('Enough!')
    time.sleep(1)
    return state, state + 1

def _filter(x):
    """

    :param x:
    :return:
    """
    print('Stage', x)
    return x + 1


def consume(x):
    """

    :param x:
    :return:
    """
    print('Consumed', x)

if __name__ == '__main__':
    p = Pipeline(produce, _filter, _filter, _filter, consume)
    initial_state = 0
    p.start(initial_state)