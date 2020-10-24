class DataAnalysisManager:

    def __init__(self):
        """

        """

        self._data = list()

    def append(self, data):
        """

        """

        self._data.append(data)

    def get_data(self, name=None):
        """

        """
        if name:
            for data in self._data:

                if name == data.get_name():

                    return data
        else:

            return [data for data in self._data]

    def get_names(self):
        """

        """
        return [_data.get_name() for _data in self._data]

    def get_types(self):
        """

        """
        return [_data._type for _data in self._data]

    def get_descriptions(self):
        """

        """
        return [_data.description for _data in self._data]

    def summary(self):
        """
        Returns a Preprocess Manager Summary (dict).
        """
        result = dict()

        names = self.get_names()

        result["length"] = len(names)
        result["names"] = names
        result["descriptions"] = self.get_descriptions()

        return result