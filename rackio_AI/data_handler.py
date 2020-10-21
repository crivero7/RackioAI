import pandas as pd
import numpy as np

class DataHandler:
    """

    """
    def __init__(self, data):
        """

        """
        self.data = data

    def insert_column(self, loc, column, data, allow_duplicates=False):
        """

        """
        if isinstance(data, np.ndarray):

            data = pd.Series(data)

        self.data.insert(loc, column, pd.Series(data), allow_duplicates=allow_duplicates)

        return self.data

    def insert_columns(self, locs, columns, data, allow_duplicates=False):
        """

        """
        if isinstance(data, np.ndarray):
            for count, column in enumerate(column_names):
                self.data[column] = data[:, count]
        elif isinstance(data, pd.DataFrame):
            pass

    def rm_column(self, *args):
        """

        """
        pass

    def rename_column(self, data, *args):
        """

        """
        pass

    def change_column(self, data, *args):
        """

        """
        pass