import pandas as pd
import numpy as np
from rackio_AI.core import RackioAI


class RackioEDA:
    """
    This is a **RackioAI** class it allows to you to handle the data embedded in **RackioAI**

    **Attributes**

    * **data:** (pd.DataFrame)
    """

    app = RackioAI()

    def __init__(self, name, description, ):
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
        return self._data

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
            self._data = pd.DataFrame(value)
        else:
            self._data = value

        self.app._data = value

    def insert_column(self, data, column, loc=None, allow_duplicates=False):
        """
        Insert column in any location in **RackioAI.data**

        ___
        **Parameters**

        * **:param data:** (np.ndarray or pd.Series) column to insert
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

        if not loc:
            loc = self.data.shape[-1]

        self.data.insert(loc, column, data, allow_duplicates=allow_duplicates)

        return self.data

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

    def remove_columns(self, *args):
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
        for column_name in args:
            self.data.pop(column_name)

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
