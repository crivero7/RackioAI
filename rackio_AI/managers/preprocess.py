from .core import Manager

class PreprocessManager(Manager):
    """
    Preprocessing object's manager

    """

    def __init__(self):

        self._obj = list()

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
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess7 = Preprocessing(name= 'Preprocess7',description='preprocess for data', problem_type='regression')
        >>> preprocess8 = Preprocessing(name= 'Preprocess8',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess7)
        >>> RackioAI.append_preprocessing_model(preprocess8)
        >>> preprocessing_manager = RackioAI.get_manager('Preprocessing')
        >>> preprocessing_types = preprocessing_manager.get_types()

        ```
        """
        return [obj._type for obj in self._obj]

    def summary(self):
        """
        Get Preprocessing Summary

        **Parameters**

        None

        **:return:**

        * **result:** (dict) key values {'length', 'names', 'descriptions', 'types'}

        ## Snippet code

        ```python
        >>> from rackio_AI import Preprocessing, RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess11 = Preprocessing(name= 'Preprocess11',description='preprocess for data', problem_type='regression')
        >>> preprocess12 = Preprocessing(name= 'Preprocess12',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess11)
        >>> RackioAI.append_preprocessing_model(preprocess12)
        >>> preprocessing_manager = RackioAI.get_manager('Preprocessing')
        >>> preprocessing_summary = preprocessing_manager.summary()

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