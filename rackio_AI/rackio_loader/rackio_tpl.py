import os
from .core import ReaderTPL
from .options import TPLOptions


class TPL:

    file_extension = ".tpl"

    def __init__(self):
        """
        This class help you to read .tpl files from OLGA simulator and convert it into dataframes, series or csv, so you
        can handle it easier to data analysis
        path: (str) it can be a filename or a directory, if it's a directory so the class will read all .tpl files from
        that parent root. if it's a filename so the TPLc class will read only that file
        return: An instance os TPL class
        """
        super(TPL, self).__init__()


    def read(self, path):
        """
        This method read the file or files in self.path
        return: A list of dictionaries with .tpl files
        """
        self.reader = ReaderTPL(path)
        self.reader.options = TPLOptions(split_expression="CATALOG")

        self.doc = self.reader()

        return self.doc


    def to(self, data_type, **kwargs):
        """
        This method allows transform from .tpl to 'data_type'
        data_type: (str) 'dataframe' - 'series' - 'csv'
        path : (str ) path to save csv file
        filename: (str) 'name.csv' if date_type == 'csv'

        return: pandas.dataframe or pandas.serie or .csv file
        """
        kwargs_default = {'path': os.getcwd(),
                          'filename': 'tpl_to_csv.csv'}
        options = {key: kwargs[key] if key in kwargs.keys() else kwargs_default[key] for key in kwargs_default.keys()}
        if data_type.lower() == 'dataframe':

            return self.reader._to_dataframe()

        elif data_type.lower() == 'series':

            return self.reader._to_series()

        elif data_type.lower() == 'csv':

            self.reader._to_csv(**options)
            return self.doc

        else:

            raise NameError('{} is not possible convert to {}'.format(type(self).__name__, data_type.lower()))