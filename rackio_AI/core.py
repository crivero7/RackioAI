import os, pickle
import pandas as pd
import numpy as np
from rackio_AI._singleton import Singleton
from rackio_AI.managers import PreprocessManager
from rackio_AI.readers import Reader
from rackio_AI.managers import DataAnalysisManager
from rackio_AI.managers import ModelsManager
from rackio_AI.preprocessing import SyntheticData


class RackioAI(Singleton):
    """
    This is the main class of the package.

    **RackioAI** is an open source, MIT License [Rackio-Framework](https://github.com/rack-io/rackio-framework) extension
    to do data analysis (based on [Pandas](https://pandas.pydata.org/)) and deep learning models (based on [Keras](https://keras.io/))
    taking advantage of **Rackio** system architecture.

    You can use it by the following snippet code:
    ```python
    >>> from rackio import Rackio
    >>> from rackio_AI import RackioAI
    >>> app = Rackio()
    >>> RackioAI(app)

    ```
    """

    def __init__(self):
        super(RackioAI, self).__init__()
        self.reader = Reader()
        self.synthetic_data = SyntheticData()
        self._preprocessing_manager = PreprocessManager()
        self._data_analysis_manager = DataAnalysisManager()
        self._models_manager = ModelsManager()
        self.app = None
        self._data = None

    def __call__(self, app):
        """

        :param app:
        :return:
        """
        self.app = app

    def load(self, filename):
        """
        You can load data in the following extensions:

        * **.tpl:** Is an [OLGA](https://www.petromehras.com/petroleum-software-directory/production-engineering-software/olga-dynamic-multiphase-flow-simulator)
        extension file.
        * **.pkl:** Numpy arrays or Pandas.DataFrame saved in pickle format.

        ___
        **Parameters**

        * **:param filename:** (str) Complete path with its extension. If the *filename* is a directory, it will load all the files
        with that extension in the directory, and if in the directory there are more directories, it will inspect it to look for more
        files with that extension.

        If the filename is a file with a valid extension, this method will load only that file.

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> filename = os.path.join(get_directory('Leak'), 'Leak112.tpl')
        >>> RackioAI.load(filename)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak112
        1            0.502732  ...  Leak112
        2            1.232772  ...  Leak112
        3            1.653696  ...  Leak112
        4            2.200430  ...  Leak112
        ...               ...  ...      ...
        3210      1617.966000  ...  Leak112
        3211      1618.495000  ...  Leak112
        3212      1619.025000  ...  Leak112
        3213      1619.554000  ...  Leak112
        3214      1620.083000  ...  Leak112
        <BLANKLINE>
        [3215 rows x 12 columns]

        **Example loading a .tpl file**

        >>> filename = os.path.join('data', 'Leak')
        >>> RackioAI.load(filename)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak112
        1            0.502732  ...  Leak112
        2            1.232772  ...  Leak112
        3            1.653696  ...  Leak112
        4            2.200430  ...  Leak112
        ...               ...  ...      ...
        35397     1618.124000  ...  Leak120
        35398     1618.662000  ...  Leak120
        35399     1619.200000  ...  Leak120
        35400     1619.737000  ...  Leak120
        35401     1620.275000  ...  Leak120
        <BLANKLINE>
        [35402 rows x 12 columns]

        **Example loading a .pkl with pandas.dataFrame**

        >>> filename = os.path.join('data', 'pkl_files', 'test_data.pkl')
        >>> RackioAI.load(filename)
                    Pipe-60 Totalmassflow_(KG/S)  ...  Pipe-151 Pressure_(PA)
        0.000000                        37.83052  ...                352683.3
        0.502732                        37.83918  ...                353449.8
        1.232772                        37.83237  ...                353587.3
        1.653696                        37.80707  ...                353654.8
        2.200430                        37.76957  ...                353706.8
        ...                                  ...  ...                     ...
        383.031800                     169.36700  ...                374582.2
        383.518200                     169.37650  ...                374575.9
        384.004500                     169.38550  ...                374572.7
        384.490900                     169.39400  ...                374573.0
        384.977200                     169.40170  ...                374576.1
        <BLANKLINE>
        [20000 rows x 4 columns]

        ```
        """
        if os.path.isdir(filename) or os.path.isfile(filename):

            if filename.endswith('.pkl'):

                try:

                    with open(filename, 'rb') as file:

                        data = pickle.load(file)

                    self.data = data

                    return data

                except:

                    with open(filename, 'rb') as file:

                        data = pd.read_pickle(file)

                    self.data = data

                    return data

            data = self._load_data(filename)
            self._data = data

            return data

        else:

            raise FileNotFoundError('{} not found'.format(filename))

    @property
    def data(self):
        """
        Variable where is storaged the loaded data.

        **Parameters**

        None

        **:return:**

        * **data:** (pandas.DataFrame)

        """
        return self._data

    @data.setter
    def data(self, value):
        """
        **Parameters**

        * **:param value:** (pd.DataFrame or np.ndarray)

        **:return:**

        None
        """
        if isinstance(value, pd.DataFrame) or isinstance(value, np.ndarray):

            if isinstance(value, np.ndarray):
                value = pd.DataFrame(value)

            self.synthetic_data.data = value
        else:
            raise TypeError('value must be a pd.DataFrame or np.ndarray')

        self._data = value

    def append_data(self, eda_object):
        """
        Append a RackioEDA object to the data analysis manager.

        ___
        **Parameters**

        * **:param eda_object:** (RackioEDA): RackioEDA object.

        **:return:**

        None

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> EDA = RackioEDA(name='EDA', description='Object Exploratory Data Analysis')
        >>> RackioAI.append_data(EDA)

        ```
        """

        self._data_analysis_manager.append(eda_object)

    def get_object(self, name, _type='EDA', serialize=False):
        """
        Get any coupled object as RackioAI attribute like *RackioEDA* object, *Preprocessing* object and *RackioDNN* object
        by its name

        ___
        **Parameters**

        * **:param name:** (str) Object name
        * **:param _type:** (str) Object type
            * **'EDA':** refers to a *DataAnalysis* or *RackioEDA* object
            * **'Preprocessing':** refers to a *Preprocessing* object
            * **'Model':** refers to a **Model** machine learning or deep learning object
        * **:param serialize:** (bool) default=False, if is True, you get a serialized object, otherwise you get the object

        **:return:**

        * **object:** (object, serialized dict)

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> EDA1 = RackioEDA(name='EDA1', description='Object 1 Exploratory Data Analysis')
        >>> RackioAI.append_data(EDA1)
        >>> eda_object = RackioAI.get_object('EDA1', _type='EDA')

        ```
        """
        if _type.lower()=='eda':

            if serialize:

                return self._serialize_data(name)

            return self._get_data(name)

        elif _type.lower()=='preprocesing':

            if serialize:

                return self._serialize_preprocessing(name)

            return self._get_preprocessing(name)
        else:
            raise TypeError('Is no possible get {} object from RackioAI'.format(_type))

        return

    def _get_data(self, name):
        """
        Returns a RackioEDA object defined by its name.

        **Parameters**

        * **:param name:** (str) RackioEDA name.

        **:return:**

        * **data_object:** (RackioEDA) RackioEDA object
        """

        return self._data_analysis_manager.get(name)

    def _serialize_data(self, name):
        """
        serialize RackioEDA

        **Parameters**

        * **:param name:** (str) RackioEDA object name


        **:return:**

        * **data:** (dict) RackioEDA object serialized
        """

        data = self.get_object(name, _type='EDA')

        return data.serialize()


    def append_preprocessing_model(self, preprocessing_model):
        """
         Append a Preprocessing object to the data analysis manager.

        ___
        **Parameters**

        * **:param preprocessing_model:** (Preprocessing): Preprocessing object.

        **:return:**

        None

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import Preprocessing, RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess1 = Preprocessing(name= 'Preprocess1',description='preprocess for data', problem_type='regression')
        >>> preprocess2 = Preprocessing(name= 'Preprocess2',description='preprocess for data', problem_type='classification')
        >>> RackioAI.append_preprocessing_model(preprocess1)
        >>> RackioAI.append_preprocessing_model(preprocess2)

        ```
        """

        self._preprocessing_manager.append(preprocessing_model)

    def get_manager(self, name):
        """
         Get a manager by its name.

        ___
        **Parameters**

        * **:param name:** (str): Manager object.
            * *'EDA'*
            * *'Preprocessing'*
            * *'Models'*

        **:return:**

        * **result:** (obj) manager object

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> EDA2 = RackioEDA(name= 'EDA2', description='Object 2 Exploratory Data Analysis')
        >>> EDA3 = RackioEDA(name= 'EDA3', description='Object 3 Exploratory Data Analysis')
        >>> RackioAI.append_data(EDA2)
        >>> RackioAI.append_data(EDA3)
        >>> eda_manager = RackioAI.get_manager('EDA')

        ```
        """
        if name.lower() == 'eda':

            result = self._data_analysis_manager
        elif name.lower() == 'preprocessing':

            result = self._preprocessing_manager

        elif name.lower() == 'models':

            result = self._models_manager

        if result:

            return result

        return

    def _get_preprocessing(self, name):
        """
        Returns a Preprocessing object defined by its name.

        **Parameters**

        * **:param name:** (str) Preprocessing name.

        **:return:**

        * **preprocessing_model:** (Preprocesing) Preprocessing model
        """

        return self._preprocessing_manager.get(name)

    def _serialize_preprocessing(self, name):
        """
        serialize Preprocessing model

        **Parameters**

        * **:param name:** (str) Preprocessing model name

        **:return:**

        * **data:** (dict) Preprocessin model serialized
        """

        preprocess = self._get_preprocessing(name)

        return preprocess.serialize()

    def summary(self):
        """
        Get a RackioAI summary.

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (dict) All defined Managers

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> EDA4 = RackioEDA(name= 'EDA4', description='Object 4 Exploratory Data Analysis')
        >>> EDA5 = RackioEDA(name= 'EDA5', description='Object 5 Exploratory Data Analysis')
        >>> RackioAI.append_data(EDA4)
        >>> RackioAI.append_data(EDA5)
        >>> eda_manager = RackioAI.get_manager('EDA')
        >>> eda_summary = eda_manager.summary()

        ```
        """

        result = dict()
        result["preprocessing manager"] = self._preprocessing_manager.summary()
        result["data analysis manager"] = self._data_analysis_manager.summary()

        return result

    @staticmethod
    def save_obj(obj, filename, format='pkl'):
        """
        Method to persist any object

        ___
        **Parameters**

        * **:param obj:** (obj) any persistable object
        * **:param filename:** (str) with no extension
        * **:param format:** (str) with no dot (.) at the beginning (default='pkl')

        **:return:**

        * obj in the path defined by *filename*
        """

        if format.lower()=='pkl':

            with open('{}.{}'.format(filename,format), 'wb') as file:
                pickle.dump(obj, file)

    @staticmethod
    def load_obj(filename, format='pkl'):
        """
        load any saved object with RackioAI's save_obj method

        ___
        **Parameters**

        * **:param filename:** (str) with no extension
        * **:param format:** (str) with no dot (.) at the beginning

        * **:return:**

        * **obj:** (obj)
        """

        obj = None
        if format.lower()=='pkl':

            with open('{}.{}'.format(filename,format), 'rb') as file:
                obj = pickle.load(file)

        return obj

    def load_test_data(self, name):
        """
        Load RackioAI test data contained in folder data

        rackio_AI package has a folder called data

        > rackio_AI/data

        In this directory there are the following folders

        > rackio_AI/data/Leak
        > rackio_AI/data/pkl_files

        *test_data* allows to you an specific file or all files in the previous folders

        ___
        **Parameters**

        * **:param name:** (str) a folder name or filename in rackio_AI/data

        * **:return:**

        * **data:** (pandas.DataFrame)
        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> directory = os.path.join('data', 'Leak')
        >>> RackioAI.load_test_data(directory) # Load test data fron a folder
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak112
        1            0.502732  ...  Leak112
        2            1.232772  ...  Leak112
        3            1.653696  ...  Leak112
        4            2.200430  ...  Leak112
        ...               ...  ...      ...
        67584     1618.124000  ...  Leak120
        67585     1618.662000  ...  Leak120
        67586     1619.200000  ...  Leak120
        67587     1619.737000  ...  Leak120
        67588     1620.275000  ...  Leak120
        <BLANKLINE>
        [67589 rows x 12 columns]

        >>> filename = os.path.join('data', 'Leak', 'Leak111.tpl')
        >>> RackioAI.load_test_data(filename) # Load test data from a file in Leak Folder
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak111
        1            0.502732  ...  Leak111
        2            1.232772  ...  Leak111
        3            1.653696  ...  Leak111
        4            2.200430  ...  Leak111
        ...               ...  ...      ...
        3214      1618.327000  ...  Leak111
        3215      1618.849000  ...  Leak111
        3216      1619.370000  ...  Leak111
        3217      1619.892000  ...  Leak111
        3218      1620.413000  ...  Leak111
        <BLANKLINE>
        [3219 rows x 12 columns]

        ```
        """

        data = self._load_data(name)

        return data

    def _load_data(self, filename):
        """

        """
        return self.reader.read(filename)

if __name__=="__main__":
    import doctest
    doctest.testmod()