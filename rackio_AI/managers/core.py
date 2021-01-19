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
        >>> from rackio_AI import RackioEDA, Preprocessing, RackioAI
        >>> EDA1 = RackioEDA(name='EDA1', description='Object 1 Exploratory Data Analysis')
        >>> EDA2 = RackioEDA(name='EDA2', description='Object 2 Exploratory Data Analysis')
        >>> preprocess1 = Preprocessing(name= 'Preprocess1',description='preprocess for data', problem_type='regression')
        >>> preprocess2 = Preprocessing(name= 'Preprocess2',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess1)
        >>> RackioAI.append_preprocessing_model(preprocess2)

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
        >>> from rackio_AI import RackioEDA, Preprocessing, RackioAI
        >>> EDA3 = RackioEDA(name='EDA3', description='Object 3 Exploratory Data Analysis')
        >>> EDA4 = RackioEDA(name='EDA4', description='Object 4 Exploratory Data Analysis')
        >>> eda_manager = RackioAI.get_manager('EDA')
        >>> EDA_objs = eda_manager.get()
        >>> EDA1_obj = eda_manager.get(name='EDA3')
        >>> preprocess3 = Preprocessing(name= 'Preprocess3',description='preprocess for data', problem_type='regression')
        >>> preprocess4 = Preprocessing(name= 'Preprocess4',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess3)
        >>> RackioAI.append_preprocessing_model(preprocess4)
        >>> preprocessing_manager = RackioAI.get_manager('Preprocessing')
        >>> preprocessing_objs = preprocessing_manager.get()
        >>> preprocessing3_obj = preprocessing_manager.get(name='Preprocess3')

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
        >>> from rackio_AI import RackioEDA, Preprocessing, RackioAI
        >>> EDA5 = RackioEDA(name= 'EDA5', description='Object 5 Exploratory Data Analysis')
        >>> EDA6 = RackioEDA(name= 'EDA6', description='Object 6 Exploratory Data Analysis')
        >>> eda_manager = RackioAI.get_manager('EDA')
        >>> eda_names = eda_manager.get_names()
        >>> preprocess5 = Preprocessing(name= 'Preprocess5',description='preprocess for data', problem_type='regression')
        >>> preprocess6 = Preprocessing(name= 'Preprocess6',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess5)
        >>> RackioAI.append_preprocessing_model(preprocess6)
        >>> preprocessing_manager = RackioAI.get_manager('Preprocessing')
        >>> preprocessing_names = preprocessing_manager.get_names()

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
        >>> from rackio_AI import RackioEDA, Preprocessing, RackioAI
        >>> EDA7 = RackioEDA(name= 'EDA7', description='Object 7 Exploratory Data Analysis')
        >>> EDA8 = RackioEDA(name= 'EDA8', description='Object 8 Exploratory Data Analysis')
        >>> eda_manager = RackioAI.get_manager('EDA')
        >>> descriptions = eda_manager.get_descriptions()
        >>> preprocess9 = Preprocessing(name= 'Preprocess9',description='preprocess for data', problem_type='regression')
        >>> preprocess10 = Preprocessing(name= 'Preprocess10',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess9)
        >>> RackioAI.append_preprocessing_model(preprocess10)
        >>> preprocessing_manager = RackioAI.get_manager('Preprocessing')
        >>> preprocessing_descriptions = preprocessing_manager.get_descriptions()

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
        >>> from rackio_AI import RackioEDA, RackioAI
        >>> EDA9 = RackioEDA(name='EDA9', description='Object 9 Exploratory Data Analysis')
        >>> EDA10 = RackioEDA(name='EDA10', description='Object 10 Exploratory Data Analysis')
        >>> eda_manager = RackioAI.get_manager('EDA')
        >>> eda_summary = eda_manager.summary()

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