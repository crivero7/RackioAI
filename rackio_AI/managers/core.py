class Manager:

    def __init__(self):

        self.obj = list()

    def append(self, obj):
        """
        Append **RackioEDA**, **Preprocessing** or **RackioDNN** object to the manager

        ___

        **Parameters**

        * **:param obj:** (RackioEDA, Preprocessing, RackioDNN) object

        **:return:**

        None

        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, Preprocessing
        >>> EDA = RackioEDA(name='EDA', description='Object Exploratory Data Analysis')
        >>> preprocess = Preprocessing(name='Preprocess', description='preprocess for data')

        ```
        """
        names = self.get_names()
        if obj.get_name() not in names:

            self.obj.append(obj)

        else:

            raise NameError('{} is already in {}'.format(obj.get_name(), self.__class__.__name__))

    def get(self, name=None):
        """
        Get a **RackioEDA**, **Preprocessing** or **RackioDNN** object by name

        ___
        **Parameters**

        * **:param name:** (str) If name is not given yo get all RackioEDA objects in the manager

        **:return:**

        * **obj:** **RackioEDA**, **Preprocessing** or **RackioDNN** object or object list

        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> manager = RackioAI.get_manager('EDA')
        >>> EDA = manager.get("EDA")
        >>> manager = RackioAI.get_manager('Preprocessing')
        >>> Preprocess = manager.get("Preprocess")

        ```
        """
        if name:
            for obj in self.obj:

                if name == obj.get_name():

                    return obj
        else:

            return [obj for obj in self.obj]

    def get_names(self):
        """
        Get all **RackioEDA**, **Preprocessing** or **RackioDNN** object names in the manager

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (list['str'])
        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> manager = RackioAI.get_manager('EDA')
        >>> manager.get_names()
        ['EDA']
        >>> manager = RackioAI.get_manager('Preprocessing')
        >>> manager.get_names()
        ['Preprocess']

        ```
        """
        return [obj.get_name() for obj in self.obj]

    def get_descriptions(self):
        """
        Get all **RackioEDA**, **Preprocessing** or **RackioDNN** object descriptions in the manager

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (list['str'])
        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> manager = RackioAI.get_manager('EDA')
        >>> manager.get_descriptions()
        ['Object Exploratory Data Analysis']
        >>> manager = RackioAI.get_manager('Preprocessing')
        >>> manager.get_descriptions()
        ['preprocess for data']

        ```
        """
        return [obj.description for obj in self.obj]

    def summary(self):
        """
        Get **RackioEDA**, **Preprocessing** or **RackioDNN** Summary

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (dict) key values {'length', 'names', 'descriptions'}

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> manager = RackioAI.get_manager('EDA')
        >>> manager.summary()
        {'length': 1, 'names': ['EDA'], 'descriptions': ['Object Exploratory Data Analysis']}
        >>> manager = RackioAI.get_manager('Preprocessing')
        >>> manager.summary()
        {'length': 1, 'names': ['Preprocess'], 'descriptions': ['preprocess for data'], 'types': ['regression']}

        ```
        """
        result = dict()

        names = self.get_names()

        result["length"] = len(names)
        result["names"] = names
        result["descriptions"] = self.get_descriptions()

        return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()