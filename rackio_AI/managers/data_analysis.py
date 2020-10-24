class DataAnalysisManager:
    """
    ...Description here...

    """

    def __init__(self):
        """
        ...Description here...
        """

        self._data = list()

    def append(self, data):
        """
        ...Description here...

        **Parameters**

        * **:param data:**

        **:return:**
        """

        self._data.append(data)

    def get_data(self, name=None):
        """
        ...Description here...

        **Parameters..

        * **:param name:**

        **:return:**

        """
        if name:
            for data in self._data:

                if name == data.get_name():

                    return data
        else:

            return [data for data in self._data]

    def get_names(self):
        """
        ...Description here...

        **Parameters**

        None

        **:return:**

        * **result:** (list)
        """
        return [_data.get_name() for _data in self._data]

    def get_types(self):
        """
        ...Description here...

        **Parameters**

        None

        **:return:**

        * **result:** (list)
        """
        return [_data._type for _data in self._data]

    def get_descriptions(self):
        """
        ...Description here...

        **Parameters**

        None

        **:return:**

        * **result:** (list)
        """
        return [_data.description for _data in self._data]

    def summary(self):
        """
        ...Description here...

        **Parameters**

        None

        **:return:**

        * **result:** (list)
        """
        result = dict()

        names = self.get_names()

        result["length"] = len(names)
        result["names"] = names
        result["descriptions"] = self.get_descriptions()

        return result