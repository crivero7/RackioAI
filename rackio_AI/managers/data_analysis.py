from .core import Manager

class DataAnalysisManager(Manager):
    """
    Data analysis object's manager

    """
    def __init__(self):

        self._obj = list()


if __name__ == "__main__":
    import doctest
    doctest.testmod()