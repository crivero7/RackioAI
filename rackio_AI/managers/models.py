class ModelsManager:
    """
    ...Documentation here...
    """

    def __init__(self):
        """
        ...Documentation here...
        """

        self._models = list()

    def append_model(self, model):
        """
        ...Documentation here...

        **Parameters**

        * **:param model:**

        **:return:**

        """

        self._models.append(model)

    def get_model(self, name):
        """
        ...Documentation here...

        **Parameters**

        * **:param name:**

        **:return:**

        """

        for _model in self._models:

            if name == _model.get_name():

                return _model

        return

    def get_models(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """

        return [model for model in self._models]

    def get_names(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return [_model.get_name() for _model in self._models]

    def get_types(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return [_model._type for _model in self._models]

    def get_descriptions(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return [_model.description for _model in self._models]

    def summary(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        result = dict()

        names = self.get_names()
        result["length"] = len(names)
        result["names"] = names
        result["descriptions"] = self.get_descriptions()
        result["types"] = self.get_types()

        return result