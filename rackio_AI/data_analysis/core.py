from ..decorators import del_temp_props
import pandas as pd
import numpy as np
from rackio_AI.core import RackioAI
from ..utils import Utils
from ..pipeline import Pipeline
from easy_deco.progress_bar import ProgressBar
import datetime
from itertools import combinations as Combina


@del_temp_props
class RackioEDA(Pipeline):
    """
    This is a **RackioAI** class it allows to you to handle the data embedded in **RackioAI**

    **Attributes**

    * **data:** (pd.DataFrame)
    """

    app = RackioAI()

    def __init__(self, name="EDA", description="EDA Pipeline"):
        super(RackioEDA, self).__init__()
        self._name = name
        self._description = description
        self._data = None

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

    @staticmethod
    def insert_column(df: pd.DataFrame, data, column_name, loc=None, allow_duplicates=False):
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
        >>> df2 = pd.DataFrame(np.array([[10], [11], [12]]), columns=['Four'])
        >>> EDA.insert_column(df2, df2.columns[0])
           One  Two  Three  Four
        0    1    2      3    10
        1    4    5      6    11
        2    7    8      9    12

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

    def insert_columns(self, data, columns, locs=[], allow_duplicates=False):
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
        >>> df2 = pd.DataFrame(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]), columns=['Four','Five','Six'])
        >>> EDA.insert_columns(df2, df2.columns.to_list())
           One  Two  Three  Four  Five  Six
        0    1    2      3    10    11   12
        1    4    5      6    13    14   15
        2    7    8      9    16    17   18

        ```
        """
        if isinstance(data, pd.DataFrame):
            data = data.values  # converting to np.ndarray

        for count, column in enumerate(columns):
            if not locs:
                self.insert_column(data[:, count], column, allow_duplicates=allow_duplicates)

            else:
                self.insert_column(data[:, count], column, locs[count], allow_duplicates=allow_duplicates)

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
        >>> EDA.remove_columns('Two', 'Three')
           One
        0    1
        1    4
        2    7

        ```
        """
        self.data = df

        self.__remove_columns(args)

        # self.del_temp_prop()

        return self.data

    def rename_columns(self, **kwargs):
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
        >>> EDA.rename_columns(**columns_to_rename)
           one  two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        >>> EDA.rename_columns(one='One',Three='three')
           One  two  three
        0    1    2      3
        1    4    5      6
        2    7    8      9

        ```
        """
        self.data = self.data.rename(columns=kwargs)

        return self.data

    def change_columns(self, data, *args):
        """
        This method allows to you rename one or several column names in the data

        ___
        **Parameters**

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
        >>> EDA.change_columns(data, *columns)
           One  Two  Three
        0    1   10     11
        1    4   13     14
        2    7   16     17

        ```
        """
        # Validate inputs
        if isinstance(data, np.ndarray):
            if data.shape[1] == len(args):
                data = pd.DataFrame(data, columns=args)

            else:
                raise ValueError('You must have the same amount of positional arguments as columns in the data')

        # Changing data
        for column in args:
            if column in self.data:
                self.data.loc[:, column] = data.loc[:, column]

        return self.data

    def search_loc(self, column_name, *keys, **kwargs):
        """
        This method allows you to rename one or several column names in the data

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
        self.data = df

        # self.del_temp_prop()

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
        self.data = df

        # self.del_temp_prop()

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

    def print_report(self, df, info=True, head=True, header=10):
        """
        Documentation here
        """
        if info:
            df.info()

        if head:
            print(df.head(header))
        
        return df

    @ProgressBar(desc="Creating combinations...", unit="dataset")
    def __combine_columns(self, combinations, **kwargs):
        """
        Documentation here
        """
        column_names = Utils.get_column_names(self.data)
        break_point = kwargs["breakpoint"]
        iloc_breakpoint = self.data.columns.get_loc(break_point)

    def combine_columns(self, df, from_columns=[], to_columns=[], **kwargs):
        """
        Documentation here
        breakpoint: column_name
        breakpoint_loc: (str) "after", "before"
        """
        self.start = 0
        self.data = df
        comb = Combina(from_columns, len(to_columns))
        
        self.__combine_columns(comb, **kwargs)

        return df

    # def del_temp_prop(self):
    #     """
    #     Documentation here
    #     """
    #     attributes = inspect.getmembers(self, lambda variable:not(inspect.isroutine(variable)))
        
    #     for variable in attributes:
            
    #         if variable[0].startswith('_') and variable[0].endswith('_'):

    #             if not(variable[0].startswith('__') and variable[0].endswith('__')):

    #                 delattr(self, variable[0])

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
