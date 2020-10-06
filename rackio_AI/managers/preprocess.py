class PreprocessManager:

    def __init__(self):
        """

        """

        self._preprocessing_models = list()

    def append_preprocessing(self, model):
        """

        """

        self._preprocessing_models.append(model)

    def get_preprocessing_model(self, name):
        """

        """

        for _model in self._preprocessing_models:
            if name == _model.get_name():
                return _model

        return

    def get_preprocessing_models(self):
        """

        """

        return [model for model in self._preprocessing_models]

    def get_names(self):
        """

        """
        return [_model.get_name() for _model in self._preprocessing_models]

    def get_types(self):
        """

        """
        return [_model._type for _model in self._preprocessing_models]

    def get_descriptions(self):
        """

        """
        return [_model.description for _model in self._preprocessing_models]


    def summary(self):
        """
        Returns a Preprocess Manager Summary (dict).
        """
        result = dict()

        names = self.get_names()

        result["length"] = len(names)
        result["names"] = names
        result["descriptions"] = self.get_descriptions()
        result["types"] = self.get_types()

        return result