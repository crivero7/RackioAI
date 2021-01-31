from .managers_core import Manager

class PreprocessManager(Manager):
    """
    Preprocessing object's manager

    """

    def __init__(self):

        super(PreprocessManager, self). __init__()

    def get_types(self):
        """
        Get all Preprocessing object types in the manager

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (list['str']) Preprocessing object types

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import Preprocessing, RackioAI
        >>> preprocess = Preprocessing(name='Preprocess', description='preprocess for data', problem_type='regression')
        >>> preprocess2 = Preprocessing(name='Preprocess2', description='preprocess for data', problem_type='classification')
        >>> manager = RackioAI.get_manager('Preprocessing')
        >>> manager.get_types()
        ['regression', 'classification']

        ```
        """
        return [obj._type for obj in self.obj]

    def summary(self):
        """
        Get Preprocessing Summary

        **Parameters**

        None

        **:return:**

        * **result:** (dict) key values {'length', 'names', 'descriptions', 'types'}

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> manager = RackioAI.get_manager('Preprocessing')
        >>> manager.summary()
        {'length': 2, 'names': ['Preprocess', 'Preprocess2'], 'descriptions': ['preprocess for data', 'preprocess for data'], 'types': ['regression', 'classification']}

        ```
        """
        result = dict()

        names = self.get_names()
        result["length"] = len(names)
        result["names"] = names
        result["descriptions"] = self.get_descriptions()
        result["types"] = self.get_types()

        return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()