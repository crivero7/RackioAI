import os
import pickle
from pickle import HIGHEST_PROTOCOL
import numpy as np
import pandas as pd
import doctest
from rackio_AI._singleton import Singleton
from rackio_AI.managers import DataAnalysisManager
from rackio_AI.managers import ModelsManager
from rackio_AI.managers import PreprocessManager
from rackio_AI.readers import Reader
from rackio_AI.utils.utils_core import Utils


class RackioAI(Singleton):
    """
    This is the main class of the package.

    **RackioAI** is an open source, MIT License [Rackio-Framework](https://github.com/rack-io/rackio-framework) extension
    to do data analysis (based on [Pandas](https://pandas.pydata.org/)) and deep learning models (based on [Keras](https://keras.io/))
    taking advantage of **Rackio** system architecture.

    You can use it by the following snippet code:
    ```python
    >>> from rackio_AI import RackioAI

    ```
    """
    def __init__(self):
        super(RackioAI, self).__init__()
        self.reader = Reader()
        self._preprocessing_manager = PreprocessManager()
        self._data_analysis_manager = DataAnalysisManager()
        self._models_manager = ModelsManager()
        self.app = None

    def __call__(self, app):
        """

        :param app:
        :return:
        """
        self.app = app

    def load(self, pathname: str, ext: str=".tpl", reset_index=False, **kwargs):
        """
        Load data into DataFrame format:

        * **.tpl:** Is an [OLGA](https://www.petromehras.com/petroleum-software-directory/production-engineering-software/olga-dynamic-multiphase-flow-simulator)
        extension file.
        * **.pkl:** Numpy arrays or Pandas.DataFrame saved in pickle format.

        ___
        **Parameters**

        * **:param pathname:** (str) Filename or directory. 
            * If the *pathname* is a directory, it will load all the files with extension *ext*.
            * If the *pathname* is a filename, it will load the file with a supported extension.
        * **:param ext:** (str) filename extension, it's necessary if pathname is a directory.
        Extensions supported are:
            * *.tpl*  [OLGA](https://www.petromehras.com/petroleum-software-directory/production-engineering-software/olga-dynamic-multiphase-flow-simulator)
        extension file.
            * *.xls*
            * *.xlsx*
            * *.xlsm*
            * *.xlsb*
            * *.odf*
            * *.ods*
            * *.odt*
            * *.csv*
            * *.pkl* (Only if the pkl saved is a DataFrame)

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> df = RackioAI.load(filename)
        >>> print(df.head())
        tag      TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                               Pressure  ...             Controller_output filename
        unit               S                                         PA  ...                                   .tpl
        0           0.000000                                   568097.3  ...                           0.0   Leak01
        1           0.502732                                   568098.2  ...                           0.0   Leak01
        2           1.232772                                   568783.2  ...                           0.0   Leak01
        3           1.653696                                   569367.3  ...                           0.0   Leak01
        4           2.200430                                   569933.5  ...                           0.0   Leak01
        <BLANKLINE>
        [5 rows x 12 columns]

        **Example loading a directory with .tpl files**

        >>> directory = os.path.join(get_directory('Leak'))
        >>> df = RackioAI.load(directory)
        >>> print(df.head())
        tag      TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                               Pressure  ...             Controller_output filename
        unit               S                                         PA  ...                                   .tpl
        0           0.000000                                   568097.3  ...                           0.0   Leak01
        1           0.502732                                   568098.2  ...                           0.0   Leak01
        2           1.232772                                   568783.2  ...                           0.0   Leak01
        3           1.653696                                   569367.3  ...                           0.0   Leak01
        4           2.200430                                   569933.5  ...                           0.0   Leak01
        <BLANKLINE>
        [5 rows x 12 columns]

        **Example loading a directory with .csv files**

        >>> directory = os.path.join(get_directory('csv'), "Hysys")
        >>> df = RackioAI.load(directory, ext=".csv", _format="hysys")
        >>> print(df.head())
          (Time, [seconds]) (PIC-118 - PV, [kPa]) (PIC-118 - OP, [%]) (SPRDSHT-1 - Cell Matrix (G-16), []) (UIC-101 - OP, [%])
        1                 0               294.769                  42                              37.6105                  10
        2               0.3               294.769                  42                              37.6105                  10
        3               0.6               294.769                  42                              37.6105                  10
        4               0.9               294.769                  42                              37.6105                  10
        5               1.1               294.769                  42                              37.6105                  10

        >>> directory = os.path.join(get_directory('csv'), "VMGSim")
        >>> df = RackioAI.load(directory, ext=".csv", _format="vmgsim")
        >>> print(df.head())
          (time, s) (/Bed-1.In.MoleFlow, kmol/h) (/Bed-1.In.P, kPa)  ... (/Sep2.In.P, kPa) (/Sep3.In.P, kPa) (/Tail_Gas.In.T, C)
        1         1                  2072.582713        285.9299038  ...       315.8859771       291.4325134                 159
        2         2                  2081.622826        286.9027793  ...       315.8953772       292.3627861                 159
        3         3                   2085.98973        287.5966429  ...       316.0995398       293.0376745                 159
        4         4                  2089.323383        288.1380485  ...       316.3974799       293.5708836                 159
        5         5                  2092.214077         288.591646  ...       316.7350299       294.0200778                 159
        <BLANKLINE>
        [5 rows x 16 columns]

        **Example loading a .pkl with pandas.dataFrame**

        >>> filename = os.path.join(get_directory('pkl_files'), 'test_data.pkl')
        >>> df = RackioAI.load(filename)
        >>> print(df.head())
           Pipe-60 Totalmassflow_(KG/S)  Pipe-151 Totalmassflow_(KG/S)  Pipe-60 Pressure_(PA)  Pipe-151 Pressure_(PA)
        0                      37.83052                       37.83052               568097.3                352683.3
        1                      37.83918                       37.70243               568098.2                353449.8
        2                      37.83237                       37.67011               568783.2                353587.3
        3                      37.80707                       37.67344               569367.3                353654.8
        4                      37.76957                       37.69019               569933.5                353706.8

        ```
        """
        filename, ext = Utils.check_path(pathname, ext=ext)

        data = self.reader.read(filename, ext=ext, **kwargs)

        self.columns_name = Utils.get_column_names(data)
        
        if data.index.has_duplicates:
        
            data = data.reset_index(drop=True)
        
        if reset_index:

            data = data.reset_index(drop=True)

        self.columns_name = Utils.get_column_names(data)

        self._data = data

        return data

    @property
    def data(self):
        """
        Variable where is storaged the loaded data.

        **Parameters**

        None

        **:return:**

        * **data:** (pandas.DataFrame)

        """
        self.columns_name = Utils.get_column_names(self._data)

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

            if hasattr(self, '_data'):

                if isinstance(value, np.ndarray):

                    self._data = pd.DataFrame(value, columns=self.columns_name)

                else:
                    
                    if isinstance(self._data.columns, pd.MultiIndex):

                        self.columns_name = pd.MultiIndex.from_tuples(self.columns_name, names=['tag', 'variable', 'unit'])

                    self._data = value
            
            else:

                self.columns_name = Utils.get_column_names(value)

                self._data = value

        else:

            raise TypeError('value must be a pd.DataFrame or np.ndarray')

    def append(self, obj):
        """
        Append a RackioEDA, Preprocessing or RackioDNN objects to managers.

        ___
        **Parameters**

        * **:param obj:** (RackioEDA, Preprocessing, RackioDNN) objects.

        **:return:**

        None

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioEDA, Preprocessing
        >>> EDA = RackioEDA(name='EDA', description='Object Exploratory Data Analysis')
        >>> Preprocess = Preprocessing(name="Preprocessing", description="Preprocesing object")

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
        Get any coupled object as RackioAI attribute like *RackioEDA*, *Preprocessing* or *RackioDNN* object
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

    def get_manager(self, _type):
        """
         Get a manager by its type.

        ___
        **Parameters**

        * **:param _type:** (str): Manager object type.
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
        if _type.lower() == 'eda':

            result = self._data_analysis_manager

        elif _type.lower() == 'preprocessing':

            result = self._preprocessing_manager

        elif _type.lower() == 'models':

            result = self._models_manager

        if result:

            return result

        return

    def summary(self):
        """
        Get RackioAI summary.

        ___
        **Parameters**

        None

        **:returns:**

        * **result:** (dict) All defined Managers

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> RackioAI.summary()
        {'preprocessing manager': {'length': 1, 'names': ['Preprocessing'], 'descriptions': ['Preprocesing object'], 'types': ['regression']}, 'data analysis manager': {'length': 1, 'names': ['EDA'], 'descriptions': ['Object Exploratory Data Analysis']}}

        ```
        """
        result = dict()
        result["preprocessing manager"] = self._preprocessing_manager.summary()
        result["data analysis manager"] = self._data_analysis_manager.summary()

        return result

    @staticmethod
    def save(obj, filename, protocol=None, format='pkl'):
        """
        Method to persist any object in pickle format

        ___
        **Parameters**

        * **:param obj:** (obj) any persistable object
        * **:param filename:** (str) with no extension
        * **:param format:** (str) with no dot (.) at the beginning (default='pkl')

        **:return:**

        * obj in the path defined by *filename*
        """
        with open('{}.{}'.format(filename, format), 'wb') as file:

            if protocol:

                pickle.dump(obj, file, protocol=protocol)

            else: 

                pickle.dump(obj, file, protocol=HIGHEST_PROTOCOL)

        return obj
