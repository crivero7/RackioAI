import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rackio_AI.core import RackioAI
from rackio_AI.utils.utils_core import Utils
from rackio_AI.pipeline import Pipeline
from rackio_AI.data_analysis.outliers import Outliers
from rackio_AI.data_analysis.noise import Noise
from easy_deco.progress_bar import ProgressBar
import datetime
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


@set_to_methods(del_temp_attr)
class RackioEDA(Pipeline):
    """
    Rackio Exploratory Data Analysis (RackioEDA for short) based on the pipe and filter
    architecture style, is an ETL framework for data extraction from homogeneous or heterogeneous
    sources, data transformation by data cleaning and transforming them into a proper storage
    format/structure for the purposes of querying and analysis; finally, data loading into the 
    final target database such as an operational data store, a data mart, data lake or a data
    warehouse.

    This schematic process is shown in the following image:

    ![ETL Process](../img/ETL.jpg)

    **Parameters**

    * **:param name:** (str) RackioEDA object's name
    * **:param description:** (str) RackioEDA object's description

    **returns**

    * **RackioEDA object**

    ```python
    >>> from rackio_AI import RackioEDA
    >>> EDA = RackioEDA(name='EDA core', description='Object Exploratory Data Analysis')

    ```
    """

    outliers = Outliers()
    noise = Noise()
    _instances = list()
    app = RackioAI()

    def __init__(self, name="", description=""):
        """
        
        """
        super(RackioEDA, self).__init__()
        self._name = name
        self._description = description
        self.app.append(self)
        RackioEDA._instances.append(self)
        
    def serialize(self):
        """
        Serialize RackioEDA object

        ___
        **Parameters**

        None

        **:return:**

        * **result:** (dict) keys {"name", "description"}

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> EDA.serialize()
        {'name': 'EDA core', 'description': 'Object Exploratory Data Analysis'}

        ```
        """
        result = {"name": self.get_name(),
                  "description": self.description}

        return result

    def get_name(self):
        """
        Get RackioEDA object's name

        **returns**

        * **name:** (str)

        ___

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> EDA.get_name()
        'EDA core'

        ```
        """
        return self._name

    @property
    def description(self):
        """
        Preprocessing attribute to storage preprocessing model description

        ___
        **Parameters**

        * **:param value:** (str) RackioEDA model description

        * **:return:**

        * **description:** (str) RackioEDA model description

        ___
        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> EDA.description
        'Object Exploratory Data Analysis'

        ```
        """
        return self._description

    @description.setter
    def description(self, value):

        self._description = value

    @property
    def data(self):
        """
        Property setter methods

        ___
        **Parameters**

        * **:param value:** (np.array, pd.DataFrame)

        **:return:**

        * **data:** (np.array, pd.DataFrame)

        ___
        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> EDA.data = df
        >>> EDA.data
           One  Two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9

        ```
        """
        return self.app.data

    @data.setter
    def data(self, value):
        """
        Property setter methods

        **Parameters**

        * **:param value:** (np.array, pd.DataFrame)

        **:return:**

        None

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> EDA.data = df
           One  Two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        ```
        """

        self.app.data = value

    def __insert_column(self, df: pd.DataFrame, data, column_name, loc=None, allow_duplicates=False):
        """
        Insert column in any location in **RackioAI.data**

        ___
        **Parameters**

        * **:param data:** (np.ndarray or pd.Series or list) column to be inserted
        * **:param column:** (str) column name to to be added
        * **:param loc:** (int) location where the column will be added, (optional, default=Last position)
        * **:param allow_duplicates:** (bool) (optional, default=False)

        **:return:**

        * **data:** (pandas.DataFrame)

        ```
        """
        if isinstance(data, np.ndarray):
            
            data = pd.Series(data)
            

        elif isinstance(data, list):

            data = pd.DataFrame(data, columns=[column_name])

        if not loc:
            
            loc = df.shape[-1]
        
        df.insert(loc, column_name, data, allow_duplicates=False)

        return df

    @ProgressBar(desc="Inserting columns...", unit="column")
    def __insert_columns(self, column_name):
        """
        Decorated function to visualize the progress bar during its execution in the pipeline

        **Parameters**

        * **:param column_name:** (list) list of data column to be inserted in DataFrame

        **returns**

        None
        """

        if not self._locs_:
            self.app.columns_name.insert(len(self.app.columns_name), column_name)
            self._data_ = self.__insert_column(self._data_, self._cols_[:, self._count_], column_name, allow_duplicates=self._allow_duplicates_)

        else:
            self.app.columns_name.insert(self._locs_, column_name)
            self._data_ = self.__insert_column(self._data_, self._cols_[:, self._count_], column_name, self._locs_[self._count_], allow_duplicates=self._allow_duplicates_)

        self._count_ += 1

        return

    def insert_columns(self, df, data, column_names, locs=[], allow_duplicates=False):
        """
        Insert columns *data* in the dataframe *df* in the location *locs*

        ___
        **Parameters**

        * **:param data:** (np.ndarray, pd.DataFrame or pd.Series) column to insert
        * **:param columns:** (list['str']) column name to to be added
        * **:param locs:** (list[int]) location where the column will be added, (optional, default=Last position)
        * **:param allow_duplicates:** (bool) (optional, default=False)

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> col = [10, 11, 12]
        >>> EDA.insert_columns(df, col, ['Four'])
           One  Two  Three  Four
        0    1    2      3    10
        1    4    5      6    11
        2    7    8      9    12

        ```
        """
        self._data_ = df
        self._locs_ = locs
        self._allow_duplicates_ = allow_duplicates
        self._count_ = 0

        if isinstance(data, list):

            data = np.array(data).reshape((-1, 1))

        elif isinstance(data, pd.DataFrame):
            
            data = data.values  # converting to np.ndarray

        self._cols_ = data

        self.__insert_columns(column_names)

        self.data = self._data_
        
        return self.data

    @ProgressBar(desc="Removing columns...", unit="column")
    def __remove_columns(self, column_name):
        """
        Decorated function to visualize the progress bar during the execution of *remove_colums*
        method in the pipeline

        **Parameters**

        * **:param column_name:** (list) list of data column to be deleted in DataFrame

        **returns**

        None
        """
        # self.app.columns_name.remove(column_name)

        self._data_ = self._data_.drop(column_name, axis=1)
        
        return

    def remove_columns(self, df, *args):
        """
        Remove columns in the data by their names

        ___
        **Parameters**

        * **:param args:** (str) column name or column names to remove from the data

        **:return:**

        * **data:** (pandas.DataFrame)
        ___
        ##Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> EDA.remove_columns(df, 'Two', 'Three')
           One
        0    1
        1    4
        2    7

        ```
        """
            
        self._data_ = df

        self.__remove_columns(args)

        self.data = self._data_

        return self._data_

    def keep_columns(self, df, *args):
        """
        Keep columns in the data by their names

        ___
        **Parameters**

        * **:param args:** (str) column name or column names to keep from the data

        **:return:**

        * **data:** (pandas.DataFrame)
        ___
        ##Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> EDA.keep_columns(df, 'Two')
           Two
        0    2
        1    5
        2    8

        ```
        """
            
        self._data_ = df

        columns = df.columns.tolist()
        [columns.remove(arg) for arg in args]

        self.__remove_columns(columns)

        self.data = self._data_

        return self._data_

    @ProgressBar(desc="Renaming columns...", unit="column")
    def __rename_columns(self, column_name, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of *rename_colums*
        method in the pipeline

        **Parameters**

        * **:param column_name:** (list) list of data column to be deleted in DataFrame

        **returns**

        None
        """
        self.data = self.data.rename(columns=kwargs)

        return

    def rename_columns(self, df, **kwargs):
        """
        Rename column names in the dataframe *df*

        ___
        **Parameters**

        * **:param df:** (pd.DataFrame) dataframe to be renamed
        * **:param kwargs:** (dict) column name or column names to remove from the data

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code
        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> columns_to_rename = {'One': 'one', 'Two': 'two'}
        >>> EDA.rename_columns(df, **columns_to_rename)
           one  two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        >>> EDA.rename_columns(df, One='one',Three='three')
           one  Two  three
        0    1    2      3
        1    4    5      6
        2    7    8      9

        ```
        """
        self.data = df
        
        self.__rename_columns(["Renaming"], **kwargs)

        return self.data

    @ProgressBar(desc="Changing columns...", unit="column")
    def __change_columns(self, column_name):
        """
        Decorated function to visualize the progress bar during the execution of *change_colums*
        method in the pipeline

        **Parameters**

        * **:param column_name:** (list) list of data column to be deleted in DataFrame

        **returns**

        None
        """
        if column_name in Utils.get_column_names(self.data):
            
            self.data.loc[:, column_name] = self._data_.loc[:, column_name]

        return

    @del_temp_attr
    def change_columns(self, df, data, column_names):
        """
        Change columns in the dataframe *df* for another columns in the dataframe *data*

        ___

        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param data:** (pandas.DataFrame) to change in *self.data*
        * **:param column_names:** (list) column or columns names to change

        **:return:**

        * **data:** (pandas.DataFrame)

        ___

        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> EDA.data = df
        >>> data = pd.DataFrame([[10, 11], [13, 14], [16, 17]], columns=['Two','Three'])
        >>> columns=['Two','Three']
        >>> EDA.change_columns(df, data, columns)
           One  Two  Three
        0    1   10     11
        1    4   13     14
        2    7   16     17

        ```
        """
        # Validate inputs
        if isinstance(data, np.ndarray):
            
            if data.shape[1] == len(column_names):
                
                data = pd.DataFrame(data, columns=column_names)

            else:
                
                raise ValueError('You must have the same amount of positional arguments as columns in the data')

        # Changing data
        self.data = df
        self._data_ = data

        self.__change_columns(column_names)

        return self.data

    def search_loc(self, column_name, *keys, **kwargs):
        """
        Logical indexing

        ___
        **Parameters**

        * **:param column_name:** (str) to change in *self.data*
        * **:param keys:** (tuple(str)) Positional arguments
        * **:param join_by:** (str)
        * **:param logic:** (str)

        **:return:**

        * **data:** (pandas.DataFrame)

        """
        default_kw = {'join_by': None,
                      'logic': '=='}

        options = {key: kwargs[key] if key in kwargs else default_kw[key] for key in default_kw}
        logics_allowed = ['==', '<', '<=', '>', '>=', '!=']

        if options['logic'] in logics_allowed:

            if options['logic'] in ['==', '=']:

                self.data = self.data[self.data.loc[:, column_name].values == keys]

            elif options['logic'] == '<':

                self.data = self.data[self.data.loc[:, column_name].values < keys]

            elif options['logic'] == '<=':

                self.data = self.data[self.data.loc[:, column_name].values <= keys]

            elif options['logic'] == '>':

                self.data = self.data[self.data.loc[:, column_name].values > keys]

            elif options['logic'] == '>=':

                self.data = self.data[self.data.loc[:, column_name].values >= keys]

            elif options['logic'] == '!=':

                self.data = self.data[self.data.loc[:, column_name].values != keys]
        else:

            raise IOError('Invalid logic')

        if options['join_by']:

            self.data.columns = self.data.columns.map(options['join_by'].join)

        return self.data

    def set_datetime_index(self, df, label, index_name, start=datetime.datetime.now(), format="%Y-%m-%d %H:%M:%S"):
        """
        Set index in dataframe *df* in datetime format

        **Parameters**

        * **:param df:** (pandas.DataFrame) Dataframe to set the index
        * **:param label:** (str) Column name that represents timeseries
        * **:param index_name:** (str) Index name
        * **:param start:** (str) datetime in string format "%Y-%m-$%d %H:%M:%S"
        * **:param format:** (str) datetime format

        **returns**

        **data** (pandas.DataFrame)
        
        ___

        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[0.5, 2, 3], [1.5, 5, 6], [3, 8, 9]], columns=['Time', 'Two', 'Three'])
        >>> df = EDA.set_datetime_index(df, "Time", "Timestamp", start="2021-01-01 00:00:00")

        ```
        """
        self._column_ = df[label].values.tolist()

        if isinstance(start, datetime.datetime):
            
            self._now_ = start

        else:

            self._now_ = datetime.datetime.strptime(start, format)

        self._timedelta_ = datetime.timedelta
        self._index_ = list()
        self._new_time_column_ = list()
        self._delta_ = list()
        self._start_ = 0

        self.__create_datetime_index(self._column_)

        df[label] = pd.DataFrame(self._new_time_column_, columns=[label])
        df.index = self._index_
        df.index.name = index_name
        self.data = df

        return df

    @ProgressBar(desc="Creating datetime index...", unit="datetime index")
    def __create_datetime_index(self, column):
        """
        Decorated function to visualize the progress bar during the execution of *set_datetime_index*
        method in the pipeline

        **Parameters**

        * **:param column:** (list) list of data column that represents timesries values

        **returns**

        None
        """
        if self._start_ == 0:
            
            self._new_time_column_.append(column)
            self._index_.append(self._now_)
            self._delta_.append(0)
            self._start_ += 1
            
            return

        self._delta_.append(column - self._column_[self._start_ - 1])

        if self._delta_[self._start_] > 0:

            self._new_time_column_.append(self._new_time_column_[self._start_ - 1] + self._delta_[self._start_])
            self._index_.append(self._index_[self._start_ - 1] + self._timedelta_(seconds=self._delta_[self._start_]))
            self._start_ += 1

        else:

            self._new_time_column_.append(self._new_time_column_[self._start_ - 1] + self._delta_[self._start_ - 1])
            self._index_.append(self._index_[self._start_ - 1] + self._timedelta_(seconds=self._delta_[self._start_ - 1]))
            self._start_ += 1

        return

    def resample(self, df, sample_time, label=None, datetime_format="%Y-%m-%d %H:%M:%S.%f", set_index=False):
        """
        Resample timeseries column in the dataframe *df*

        **Parameters**

        * **:param df:** (pandas.DataFrame) 
        * **:param sample_time:** (float or int) new sample time in the dataframe
        * **:param label:** (str) column name that represents timeseries values

        **returns**

        **data:** (pandas.DataFrame) 
         
         ___

        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[0.5, 2, 3], [1, 5, 6], [1.5, 8, 9], [2, 8, 9]], columns=['Time', 'Two', 'Three'])
        >>> EDA.resample(df, 1, label="Time")
           Time  Two  Three
        0   0.5    2      3
        2   1.5    8      9

        ```

        ```python
        >>> import pandas as pd
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([["2021-03-24 17:27:11.0", 2, 3], ["2021-03-24 17:27:11.5", 5, 6], ["2021-03-24 17:27:12.0", 8, 9], ["2021-03-24 17:27:12.5", 8, 9]], columns=['Time', 'Two', 'Three'])
        >>> EDA.resample(df, 1, label="Time")
                            Time  Two  Three
        0  2021-03-24 17:27:11.0    2      3
        2  2021-03-24 17:27:12.0    8      9

        ```
        """
        self._rows_to_delete_ = list()
        self._rows_to_keep_ = list()
        self._datetime = False
        self._diff_ = self._diff = self._row_ = 0
        label_index = 'index'
        if not label:
            if df.index.name:
                label_index = df.index.name
            label = label_index
        else:
            if df.index.name:
                label_index = df.index.name
        df = df.reset_index()
        self._column_ = df[label].values
        if isinstance(self._column_[0], (str, np.datetime64)):
            base_time = pd.to_datetime(self._column_[0], format=datetime_format)
            self._column_ = pd.to_datetime(self._column_, format=datetime_format)
            self._column_ = (self._column_ - base_time).total_seconds()
            self._datetime = True
        self._column_ = self._column_.tolist()
        options = {"freq": sample_time}
        self.__resample(self._column_, **options)
        df = df.drop(self._rows_to_delete_)
        if set_index:
            df = df.set_index(label)
        else:
            df = self.remove_columns(df, label_index)
        self.data = df
        return df

    @ProgressBar(desc="Resampling...", unit="Sampling")
    def __resample(self, column: list, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of *resample*
        method in the pipeline

        **Parameters**

        * **:param column:** (list) list of data column that represents timesries values
        * **:kwargs freq:** (float or int) new sample time in the dataframe

        **returns**

        None
        """
        freq = kwargs["freq"]
        if self._row_ == 0:
            
            self._rows_to_keep_.append(self._row_)
            self._row_ += 1
            
            return
        
        delta = column - self._column_[self._row_ - 1]
        self._diff += delta 
        self._diff_ =  freq * len(self._rows_to_keep_)

        if len(self._rows_to_keep_) == 1:

            if self._diff < freq:

                self._rows_to_delete_.append(self._row_)
                self._row_ += 1
                
                return
            
            self._rows_to_keep_.append(self._row_)
            self._row_ += 1

            return

        if abs(self._diff_ - self._diff) < freq:
            
            self._rows_to_delete_.append(self._row_)
            self._row_ += 1
            
            return
        
        self._rows_to_keep_.append(self._row_)
        # self._diff_ = 0
        self._row_ += 1

        return

    @ProgressBar(desc="Reseting index...", unit="index")
    def __reset_index(self, flag=[0]):
        """
        Decorated function to visualize the progress bar during the execution of *reset_index*
        method in the pipeline

        **Parameters**

        * **:param flag:** (list)

        **returns**

        None
        """
        return

    def reset_index(self, df: pd.DataFrame, drop: bool=False):
        """
        Reset index in the dataframe *df*

        **Parameters**

        * **:param df:** (pandas.DataFrame) 
        * **:param drop:** (bool) drop index from the dataframe

        **returns**

        **data:** (pandas.DataFrame) 
         
         ___

        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[0.5, 2, 3], [1, 5, 6], [1.5, 8, 9], [2, 8, 9]], columns=['Time', 'Two', 'Three'])
        >>> EDA.reset_index(df, drop=False)
           index  Time  Two  Three
        0      0   0.5    2      3
        1      1   1.0    5      6
        2      2   1.5    8      9
        3      3   2.0    8      9

        ```
        """
        df = df.reset_index(drop=drop)
        self.data = df
        self.__reset_index([0])

        return df

    @ProgressBar(desc="Printing report...", unit="dataframe")
    def __print_report(self, iterable):
        """
        Decorated function to visualize the progress bar during the execution of *print_report*
        method in the pipeline

        **Parameters**

        * **:param iterable:** (list)

        **returns**

        None
        """
        if self._info_:
            
            self.data.info()

        if self._head_:
            
            print(self.data.head(self._header_))

        return

    def print_report(
        self, 
        df: pd.DataFrame, 
        info: bool=True, 
        head: bool=True, 
        header: int=10
        ):
        """
        Print DataFrame report, info and head report

        ___
        **Parameters**

        * **:param df:** (pd.DataFrame) DataFrame to print report
        * **:param info:** (bool) get info from DataFrame
        * **:param head:** (bool) get head from DataFrame
        * **:param header:** (int) number of first rows to print

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code

        ```python
        >>> import pandas as pd
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['One', 'Two', 'Three'])
        >>> df = EDA.print_report(df, info=True, head=False)
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   One     3 non-null      int64
         1   Two     3 non-null      int64
         2   Three   3 non-null      int64
        dtypes: int64(3)
        memory usage: 200.0 bytes
        
        ```
        """

        self.data = df
        self._info_ = info
        self._head_ = head
        self._header_ = header
        
        self.__print_report(["Printing"])
        
        return self.data

    def fixnan(
        self, 
        df: pd.DataFrame, 
        key: str="median", 
        neighbors: int=3, 
        _round: bool=False,
        down: bool=False,
        decimals: int=5
        )->pd.DataFrame:
        """
        Fixes nan in dataframe columns by a key function

        **Parameters**
        
        * **:param df:** (pandas.DataFrame)
        * **:param key:** (str) Function's name to fix nan
            * *median*
            * *mean*
            * *std*
            * *var*
        * **:param neighbors:** (int) neighbors values to apply key function
        * **:param _round:** (bool) 
            * If True the value fixed is rounded
        * **:param down:** (bool) round down if *_round* is True otherwise round up
        * **:param decimals:** (int): If *_round* is True, the value is rounded with
        these decimals

        **returns**

        * **df** (pandas.DataFrame) dataframe with nan values fixed
        ___

        ### **Snippet code

        ```python
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> EDA = RackioAI.get(name="EDA core", _type='EDA')
        >>> df = pd.DataFrame(np.random.randn(10, 3), index=['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k'], columns=['one', 'two', 'three'])
        >>> df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
        >>> df_fixed = EDA.fixnan(df2, _round=True)

        ```
        """
        if key.lower() in  ["median", "mean", "std", "var"]:
            
            self._dict_nonan_ = dict()
            self._df_ = df
            columns = Utils.get_column_names(df)
            options = {
                "key": key,
                "neighbors": neighbors,
                "_round": _round,
                "down": down,
                "decimals": decimals
            }
            self.__first_step_fixnan(columns, **options)
        
            names = list(self._dict_nonan_.keys())
            self.__last_step_fixnan(names)

            return self._df_
        
        else:

            raise TypeError("{} is key not valid, use: ['median', 'mean', 'std', 'var']".format(key))

    @ProgressBar(desc="Analyzing dataframe...", unit="rows")
    def __first_step_fixnan(self, column, **kwargs):
        """
        Documentation here
        """
        if (self._df_[column].dtype == np.float64) & (self._df_[column].isnull().values.any()):
            # save null values index
            indexnulls = list(self._df_[column][self._df_[column].isnull()].index)
            # convert data to numpy array because it's faster
            self._data_ = self._df_[column].to_numpy()
            # save null values locations
            locs = np.where(np.isnan(self._data_))[0]

            self.__second_step_fixnan(locs, **kwargs)

            self._dict_nonan_.update(
                {
                    column:np.vstack((indexnulls, self._data_[locs]))
                }
            )
        
        return

    @ProgressBar(desc="Analyzing dataframe...", unit="columns")
    def __second_step_fixnan(self, loc, **kwargs):
        """
        Documentation here
        """
        key = kwargs["key"]
        neighbors = kwargs["neighbors"]
        _round = kwargs["_round"]
        down = kwargs["down"]
        decimals = kwargs["decimals"]

        fix = getattr(np, "nan{}".format(key))
        value = fix(self._data_[loc - neighbors: loc + neighbors])
        if _round:

            self._data_[loc] = Utils.round(value, decimals=decimals, down=down)

        else:
            
            self._data_[loc] = value

        return

    @ProgressBar(desc="Fixing nan in dataframe...", unit="columns")
    def __last_step_fixnan(self, column):
        """
        Documentatio here
        """
        _serie = pd.Series(self._dict_nonan_[column][1],
                          index=self._dict_nonan_[column][0])
        self._df_.loc[:, column].update(_serie)

        return

if __name__=="__main__":
    
    import doctest

    doctest.testmod()
