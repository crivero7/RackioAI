import os

import numpy as np
import pandas as pd

from rackio_AI._singleton import Singleton
from rackio_AI.managers import DataAnalysisManager
from rackio_AI.managers import ModelsManager
from rackio_AI.managers import PreprocessManager
from rackio_AI.readers import Reader
from rackio_AI.utils import Utils


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

    def load(self, filename: str, ext: str=".tpl", **kwargs):
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
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.load(filename)
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        3214      1618.327000                                   569341.1  ...                           0.0   Leak01
        3215      1618.849000                                   569341.3  ...                           0.0   Leak01
        3216      1619.370000                                   569341.5  ...                           0.0   Leak01
        3217      1619.892000                                   569341.7  ...                           0.0   Leak01
        3218      1620.413000                                   569341.9  ...                           0.0   Leak01
        <BLANKLINE>
        [3219 rows x 12 columns]

        **Example loading a directory with .tpl files**

        >>> directory = os.path.join(get_directory('Leak'))
        >>> RackioAI.load(directory)
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        6429      1617.966000                                   569342.5  ...                           0.0   Leak02
        6430      1618.495000                                   569342.8  ...                           0.0   Leak02
        6431      1619.025000                                   569343.0  ...                           0.0   Leak02
        6432      1619.554000                                   569343.2  ...                           0.0   Leak02
        6433      1620.083000                                   569343.4  ...                           0.0   Leak02
        <BLANKLINE>
        [6434 rows x 12 columns]

        **Example loading a .pkl with pandas.dataFrame**

        >>> filename = os.path.join(get_directory('pkl_files'), 'test_data.pkl')
        >>> RackioAI.load(filename)
               Pipe-60 Totalmassflow_(KG/S)  Pipe-151 Totalmassflow_(KG/S)  Pipe-60 Pressure_(PA)  Pipe-151 Pressure_(PA)
        0                          37.83052                       37.83052               568097.3                352683.3
        1                          37.83918                       37.70243               568098.2                353449.8
        2                          37.83237                       37.67011               568783.2                353587.3
        3                          37.80707                       37.67344               569367.3                353654.8
        4                          37.76957                       37.69019               569933.5                353706.8
        ...                             ...                            ...                    ...                     ...
        19995                     169.36700                      169.40910               784411.5                374582.2
        19996                     169.37650                      169.41690               784381.0                374575.9
        19997                     169.38550                      169.42340               784363.6                374572.7
        19998                     169.39400                      169.42930               784362.2                374573.0
        19999                     169.40170                      169.43530               784374.4                374576.1
        <BLANKLINE>
        [20000 rows x 4 columns]

        ```
        """
        filename, ext = Utils.check_path(filename, ext=ext)

        data = self.reader.read(filename, ext=ext, **kwargs)
            
        self.data = data

        return self.data

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
        else:

            raise TypeError('value must be a pd.DataFrame or np.ndarray')

        if value.index.has_duplicates:
        
            value = value.reset_index(drop=True)

        self._data = value

    def append(self, obj):
        """
        Append a RackioEDA object to managers.

        ___
        **Parameters**

        * **:param eda_object:** (RackioEDA, Preprocessing, RackioDNN) objects.

        **:return:**

        None

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, Preprocessing
        >>> EDA = RackioEDA(name='EDA', description='Object Exploratory Data Analysis')
        >>> Preprocess = Preprocessing(name="Preprocess", description="Preprocesing object")

        ```
        """
        if "RackioEDA" in str(type(obj)):
        
            self._data_analysis_manager.append(obj)
        
        elif "Preprocessing" in str(type(obj)):

            self._preprocessing_manager.append(obj)

        elif "RackioDNN" in str(type(obj)):

            pass

    def get(self, name, _type='EDA', serialize=False):
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
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get("EDA", _type="EDA")
        >>> Preprocess = RackioAI.get("Preprocess", _type="Preprocessing")

        ```
        """
        if _type.lower() == 'eda':

            data = self._data_analysis_manager.get(name)

            if serialize:
                
                return data.serialize()

            return data

        elif _type.lower() == 'preprocessing':

            preprocess = self._preprocessing_manager.get(name)

            if serialize:

                return preprocess.serialize()

            return preprocess
        
        else:
            
            raise TypeError('Is not possible get {} object from RackioAI'.format(_type))

        return

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
        >>> from rackio_AI import RackioAI
        >>> eda_manager = RackioAI.get_manager("EDA")
        >>> preprocessing_manager = RackioAI.get_manager("Preprocessing")

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
        >>> from rackio_AI import RackioAI
        >>> RackioAI.summary()
        {'preprocessing manager': {'length': 1, 'names': ['Preprocess'], 'descriptions': ['Preprocesing object'], 'types': ['regression']}, 'data analysis manager': {'length': 1, 'names': ['EDA'], 'descriptions': ['Object Exploratory Data Analysis']}}

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

        if format.lower() == 'pkl':

            with open('{}.{}'.format(filename, format), 'wb') as file:

                pickle.dump(obj, file, protocol=4)

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
        >>> from rackio_AI import RackioAI, get_directory
        >>> directory = os.path.join(get_directory('Leak'))
        >>> RackioAI.load_test_data(directory) # Load test data fron a folder
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        6429      1617.966000                                   569342.5  ...                           0.0   Leak02
        6430      1618.495000                                   569342.8  ...                           0.0   Leak02
        6431      1619.025000                                   569343.0  ...                           0.0   Leak02
        6432      1619.554000                                   569343.2  ...                           0.0   Leak02
        6433      1620.083000                                   569343.4  ...                           0.0   Leak02
        <BLANKLINE>
        [6434 rows x 12 columns]

        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.load_test_data(filename) # Load test data from a file in Leak Folder
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        3214      1618.327000                                   569341.1  ...                           0.0   Leak01
        3215      1618.849000                                   569341.3  ...                           0.0   Leak01
        3216      1619.370000                                   569341.5  ...                           0.0   Leak01
        3217      1619.892000                                   569341.7  ...                           0.0   Leak01
        3218      1620.413000                                   569341.9  ...                           0.0   Leak01
        <BLANKLINE>
        [3219 rows x 12 columns]

        ```
        """

        data = self.load(name)

        return data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
