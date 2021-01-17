import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rackio_AI.core import RackioAI
from rackio_AI.utils import Utils
from rackio_AI.pipeline import Pipeline
from easy_deco.progress_bar import ProgressBar
import datetime


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

    """

    app = RackioAI()

    def __init__(self, name="EDA", description="EDA Pipeline"):
        """
        **Parameters**

        * **:param name:** (str) RackioEDA object's name
        * **:param description:** (str) RackioEDA object's description

        **returns**

        * **RackioEDA object**
        """
        super(RackioEDA, self).__init__()
        self._name = name
        self._description = description
        
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
        >>> from rackio_AI import RackioAI, RackioEDA
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.serialize()
        {'name': 'EDA', 'description': 'Object Exploratory Data Analysis'}

        ```
        """
        result = {"name": self.get_name(),
                  "description": self.description}

        return result

    def get_name(self):
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
        >>> from rackio_AI import RackioAI, RackioEDA
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
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
        >>> import numpy as np
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df
        >>> EDA.data
           One  Two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9

        ```
        """
        return self.app._data

    @data.setter
    def data(self, value):
        """
        Property setter methods

        **Parameters**

        * **:param value:** (np.array, pd.DataFrame)

        **:return:**

        None

        ```python
        >>> import numpy as np
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df
           One  Two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        ```
        """

        if isinstance(value, np.ndarray):
            self.app._data = pd.DataFrame(value)
        else:
            self.app._data = value

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
        Documentation here
        """
        if not self._locs_:
                
            self.data = self.__insert_column(self.data, self._data_[:, self._count_], column_name, allow_duplicates=self._allow_duplicates_)

        else:

            self.data = self.__insert_column(self.data, self._data_[:, self._count_], column_name, self._locs_[self._count_], allow_duplicates=self._allow_duplicates_)

        self._count_ += 1

        return

    def insert_columns(self, df, data, column_names, locs=[], allow_duplicates=False):
        """
        Insert several columns in any location

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
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df1
        >>> df2 = [10, 11, 12]
        >>> EDA.insert_columns(df1, df2, ['Four'])
           One  Two  Three  Four
        0    1    2      3    10
        1    4    5      6    11
        2    7    8      9    12

        ```
        """
        self.data = df
        self._locs_ = locs
        self._allow_duplicates_ = allow_duplicates
        self._count_ = 0

        if isinstance(data, list):

            data = np.array(data).reshape((-1, 1))

        elif isinstance(data, pd.DataFrame):
            
            data = data.values  # converting to np.ndarray

        self._data_ = data

        self.__insert_columns(column_names)

        return self.data

    @ProgressBar(desc="Removing columns...", unit="column")
    def __remove_columns(self, columns):
        """
        Documentation here
        """
        self.data.pop(columns)
        
        return

    def remove_columns(self, df, *args):
        """
        This method allows to you remove one or several columns in the data

        ___
        **Parameters**

        * **:param args:** (str) column name or column names to remove from the data

        **:return:**

        * **data:** (pandas.DataFrame)
        ___
        ##Snippet code

        ```python
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df1
        >>> EDA.remove_columns(df1, 'Two', 'Three')
           One
        0    1
        1    4
        2    7

        ```
        """
        self.data = df

        self.__remove_columns(args)

        return self.data

    @ProgressBar(desc="Renaming columns...", unit="column")
    def __rename_columns(self, columns, **kwargs):
        """
        Documentation here
        """
        self.data = self.data.rename(columns=kwargs)

        return

    def rename_columns(self, df, **kwargs):
        """
        This method allows to you rename one or several column names in the data

        ___
        **Parameters**

        * **:param kwargs:** (dict) column name or column names to remove from the data

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code
        ```python
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df1
        >>> columns_to_rename = {'One': 'one', 'Two': 'two'}
        >>> EDA.rename_columns(df1,**columns_to_rename)
           one  two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        >>> EDA.rename_columns(df1, One='one',Three='three')
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
        Documentation here
        """
        if column_name in self.data.columns.to_list():
            
            self.data.loc[:, column_name] = self._data_.loc[:, column_name]

        return

    def change_columns(self, df, data, column_names):
        """
        This method allows to you change columns data for another columns data in a daaframe

        ___
        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param data:** (pandas.DataFrame) to change in *self.data*
        * **:param args:** (str) column or columns names to change

        **:return:**

        * **data:** (pandas.DataFrame)

        ___
        ## Snippet code
        ```python
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df1
        >>> data = np.array([[10, 11], [13, 14], [16, 17]])
        >>> columns=['Two','Three']
        >>> EDA.change_columns(df1, data, columns)
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

        ___
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

    def set_datetime_index(self, df, label):
        """

        """
        self._column_ = df[label].values.tolist()
        self._now_ = datetime.datetime.now
        self._timedelta_ = datetime.timedelta
        self._index_ = list()
        self._new_time_column_ = list()
        self._delta_ = list()
        self._start_ = 0

        self.__create_datetime_index(self._column_)

        df[label] = pd.DataFrame(self._new_time_column_, columns=[label])
        df.index = self._index_
        df.index.name = "Timestamp"

        return df

    @ProgressBar(desc="Creating datetime index...", unit="datetime index")
    def __create_datetime_index(self, column):
        """

        """
        if self._start_ == 0:
            self._new_time_column_.append(column)
            self._index_.append(self._now_())
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

    def resample(self, df, freq, label):
        """

        """
        self._rows_to_delete_ = list()
        self._diff_ = self._start_ = 0
        self._column_ = df.loc[:, label].values.reshape(1, -1).tolist()[0]
        options = {"freq": freq}
        self.__resample(self._column_, **options)
        df = df.drop(self._rows_to_delete_)

        return df

    @ProgressBar(desc="Resampling...", unit="Sampling")
    def __resample(self, column, **kwargs):
        """

        """
        freq = kwargs["freq"]

        if self._start_ == 0:
            self._start_ += 1
            return

        delta = column - self._column_[self._start_ - 1]
        self._diff_ += delta

        if abs(self._diff_) < freq:
            self._rows_to_delete_.append(self._start_)
            self._start_ += 1
            return

        self._diff_ = 0
        self._start_ += 1

        return

    @ProgressBar(desc="Reseting index...", unit="index")
    def __reset_index(self, flag=[0]):
        """
        Documentation here
        """
        return

    def reset_index(self, df, drop=False):
        """
        Documentation here
        """
        df = df.reset_index(drop=drop)
        self.data = df
        self.__reset_index([0])

        return df

    @ProgressBar(desc="Printing report...", unit="dataframe")
    def __print_report(self, iterable):
        """
        Documentation here
        """
        if self._info_:
            
            self.data.info()

        if self._head_:
            
            print(self.data.head(self._header_))

        return

    def print_report(self, df: pd.DataFrame, info=True, head=True, header=10):
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
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> EDA = RackioEDA(name= 'EDA', description='Object Exploratory Data Analysis')
        >>> EDA.data = df1
        >>> EDA.print_report(df1, info=True, head=True, header=2)
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   One     3 non-null      int32
         1   Two     3 non-null      int32
         2   Three   3 non-null      int32
        dtypes: int32(3)
        memory usage: 164.0 bytes
           One  Two  Three
        0    1    2      3
        1    4    5      6
           One  Two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9

        ```
        """

        self.data = df
        self._info_ = info
        self._head_ = head
        self._header_ = header
        
        self.__print_report(["Printing"])
        
        return self.data


class Plot:
    """
    Documentation here
    """
    def __init__(self):
        """
        Documentation
        """
        pass

if __name__ == "__main__":
    import doctest

    doctest.testmod()
