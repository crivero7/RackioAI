class PreprocessManager:
    """
    ...Documentation here...
    """

    def __init__(self):
        """
        ...Documentation here...
        """

        self._preprocessing_models = list()

    def append_preprocessing(self, model):
        """
        ...Documentation here...

        **Parameters**

        * **:param model:**

        **:return:**

        """

        self._preprocessing_models.append(model)

    def get_preprocessing_model(self, name):
        """
        ...Documentation here...

        **Parameters**

        * **:param name:**

        **:return:**

        """

        for _model in self._preprocessing_models:
            if name == _model.get_name():
                return _model

        return

    def get_preprocessing_models(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """

        return [model for model in self._preprocessing_models]

    def get_names(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return [_model.get_name() for _model in self._preprocessing_models]

    def get_types(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return [_model._type for _model in self._preprocessing_models]

    def get_descriptions(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return [_model.description for _model in self._preprocessing_models]


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