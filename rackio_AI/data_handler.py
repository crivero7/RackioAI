import pandas as pd
import numpy as np

class DataHandler:
    """
    This is a **RackioAI** class it allows to you to handle the data embedded in **RackioAI**
    **Attributes**
        * **data:** (pd.DataFrame)
    """

    def __init__(self, notify):

        self._data = None
        self.notify = notify

    @property
    def data(self):
        """
        Property getter method
        return:
            data (np.array, pd.DataFrame)
        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Property setter methods
        **parameters**
            * **value:** (np.array, pd.DataFrame)

        **return:** None

        >>> import numpy as np
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> RackioAI.data
           One  Two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        """

        if isinstance(value, np.ndarray):
            self._data = pd.DataFrame(value)
        else:
            self._data = value

        self.notify(self._data)

    def insert_column(self, data, column, loc=None, allow_duplicates=False):
        """
        Insert column in any location in **RackioAI.data**
        **Parameters**
            * **data:** (np.ndarray or pd.Series) column to insert
            * **column:** (str) column name to to be added
            * **loc:** (int) location where the column will be added, (optional, default=Last position)
            * **allow_duplicates:** (bool) (optional, default=False)
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> RackioAI.data = df1
        >>> df2 = pd.DataFrame(np.array([[10], [11], [12]]), columns=['Four'])
        >>> RackioAI.data_handler.insert_column(df2, df2.columns[0])
           One  Two  Three  Four
        0    1    2      3    10
        1    4    5      6    11
        2    7    8      9    12
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if not loc:
            loc = self.data.shape[-1]

        self.data.insert(loc, column, data, allow_duplicates=allow_duplicates)

        return self.data

    def insert_columns(self, data, columns, locs=[], allow_duplicates=False):
        """
        Insert several columns in any location in **RackioAI.data**
        **Parameters**
            * **data:** (np.ndarray, pd.DataFrame or pd.Series) column to insert
            * **columns:** (list['str']) column name to to be added
            * **locs:** (list[int]) location where the column will be added, (optional, default=Last position)
            * **allow_duplicates:** (bool) (optional, default=False)
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> RackioAI.data = df1
        >>> df2 = pd.DataFrame(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]), columns=['Four','Five','Six'])
        >>> RackioAI.data_handler.insert_columns(df2, df2.columns.to_list())
           One  Two  Three  Four  Five  Six
        0    1    2      3    10    11   12
        1    4    5      6    13    14   15
        2    7    8      9    16    17   18
        """
        if isinstance(data, pd.DataFrame):
            data = data.values  #converting to np.ndarray

        for count, column in enumerate(columns):
            if not locs:
                self.insert_column(data[:, count], column, allow_duplicates=allow_duplicates)

            else:
                self.insert_column(data[:, count], column, locs[count], allow_duplicates=allow_duplicates)

        return self.data

    def remove_columns(self, *args):
        """
        This method allows to you remove one or several columns in the data
        **Parameters**
            * **column_names:** (str) column name or column names to remove from the data
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> RackioAI.data = df1
        >>> RackioAI.data_handler.remove_columns('Two', 'Three')
           One
        0    1
        1    4
        2    7
        """
        for column_name in args:
            self.data.pop(column_name)

        return self.data

    def rename_columns(self, **kwargs):
        """
        This method allows to you rename one or several column names in the data
        **Parameters**
            * **column_names:** (dict) column name or column names to remove from the data
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> RackioAI.data = df1
        >>> columns_to_rename = {'One': 'one', 'Two': 'two'}
        >>> RackioAI.data_handler.rename_columns(**columns_to_rename)
           one  two  Three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        >>> RackioAI.data_handler.rename_columns(one='One',Three='three')
           One  two  three
        0    1    2      3
        1    4    5      6
        2    7    8      9
        """
        self.data = self.data.rename(columns=kwargs)

        return self.data

    def change_columns(self, data, *args):
        """
        This method allows to you rename one or several column names in the data
        **Parameters**
            * **data:** (dict) column name or column names to remove from the data
        >>> import pandas as pd
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['One', 'Two', 'Three'])
        >>> RackioAI.data = df1
        >>> data = np.array([[10, 11], [13, 14], [16, 17]])
        >>> columns=['Two','Three']
        >>> RackioAI.data_handler.change_columns(data, *columns)
           One  Two  Three
        0    1   10     11
        1    4   13     14
        2    7   16     17
        """
        # Validate inputs
        if isinstance(data, np.ndarray):
            if data.shape[1] == len(args):
                data = pd.DataFrame(data, columns= args)

            else:
                raise ValueError('You must have the same amount of positional arguments as columns in the data')

        # Changing data
        for column in args:
            if column in self.data:
                self.data.loc[:, column] = data.loc[:, column]

        return self.data

if __name__=="__main__":
    import doctest
    doctest.testmod()