import os
import numpy as np
import pandas as pd
from rackio_AI.decorators.progress_bar import progressBar
from .options import TPLOptions


class TPL:

    tpl_options = TPLOptions()

    def __init__(self):
        """
        This class help you to read .tpl files from OLGA simulator and convert it into dataframes, series or csv, so you
        can handle it easier to data analysis
        path: (str) it can be a filename or a directory, if it's a directory so the class will read all .tpl files from
        that parent root. if it's a filename so the TPLc class will read only that file
        return: An instance os TPL class
        """
        self.file_extension = ".tpl"
        self.doc = list()


    def read(self, filename, specific_file=True):
        """
        This method read the file or files in self.path
        return: A list of dictionaries with .tpl files
        """
        if specific_file:

            self.doc = [self._read_file(filename)]

        else:

            self.doc = self._read_all_files(filename)

        return self.doc

    def _read_file(self, filename):
        """

        """
        doc = dict()
        with open(filename, 'r') as file:
            self.file = file.read()

        (data_header_section, data) = self._get_section_from()
        header_section = self._get_header_section(data_header_section)
        (filename, _) = os.path.splitext(filename)

        multi_index = list()

        for count, column_name in enumerate(header_section):
            column_name = self._clean_column_name(column_name)

            (tag, unit, variable_type) = self._get_structure(column_name)
            multi_index.append((tag, variable_type, unit))
            "fill dictionary"
            doc[tag] = {'variable': variable_type,
                        'unit': unit,
                        'data': data[:, count]}
        data_name = np.array([filename.split(os.path.sep)[-1]] * data.shape[0])

        multi_index.append(('file', 'filename', '.tpl'))
        doc['file'] = {'variable':"file",
                       'unit': ".tpl",
                       'data': data_name}

        self.header = pd.MultiIndex.from_tuples(multi_index, names=['tag', 'variable', 'unit'])
        return doc

    def _read_all_files(self, directory):
        """

        """
        filenames = self.find_files(self.tpl_options.file_extension, directory)

        doc = self._read_files(filenames)

        self.doc = doc

        return doc

    @progressBar(desc='Loading files...', unit='files')
    def _read_files(self, filenames):
        """

        """

        self.doc.append(self._read_file(filenames))

        return self.doc

    def _get_section_from(self):
        """

        """
        sections = self.file.split("{} \n".format(self.tpl_options.split_expression))

        self.tpl_options.header_line_numbers = int(sections[1].split('\n')[0])

        data_header_section = sections[1].split('\n')

        data= self._get_data(data_header_section[self.tpl_options.header_line_numbers + 2::])

        return (data_header_section, data)

    def _get_header_section(self, data_header_section):

        """

        """
        header_section = data_header_section[1: self.tpl_options.header_line_numbers + 2]

        return header_section[-1:] + header_section[:-1]

    def _get_data(self, data):
        """

        """
        rows = len(data)
        new_data = list()

        for count, d in enumerate(data):
            if count == rows -1:
                break

            new_data.append(np.array([float(item) for item in d.split(" ")]))

        return np.array(new_data)

    @staticmethod
    def find_files(extension, path):
        """

        """
        result = list()

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    result.append(os.path.join(root, file))

        return result

    def _clean_column_name(self, column_name):
        """

        """

        return column_name.replace("'", "").replace(":", "").replace(" ", "_").replace("-", "").replace("__","_")

    def _get_tag(self, column_name):
        """

        """
        tag = column_name[0:column_name.find("(") - 1]

        if tag.endswith("_"):
            tag = tag[0:-1]

        return tag

    def _get_unit(self, column_name):
        """

        """
        return column_name[column_name.find("(") + 1:column_name.find(")")]

    def _get_variable_type(self, column_name):
        """

        """
        return column_name[column_name.find(")")+2::]

    def _get_structure(self, column_name):
        """

        """
        tag = self._get_tag(column_name)

        unit = self._get_unit(column_name)

        variable_type = self._get_variable_type(column_name)

        return (tag, unit, variable_type)

    def _to_dataframe(self):
        """

        """
        return self._join(flag=True)

    def _to_series(self):
        """

        """
        return self._join(flag=False)

    def _to_csv(self, **kwargs):
        """

        """
        df = self._join(flag=True)

        df.to_csv(os.path.join(kwargs['path'], kwargs['filename']))

    @staticmethod
    def _coerce_df_columns_to_numeric(df, column_list):

        df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
        return df

    def _join(self, flag=True):
        """

        """
        columns = self.doc[0].keys()
        if flag:

            [setattr(self, key, list()) for key in columns]

            for data in self.doc:

                for key in columns:
                    attr = getattr(self, key)
                    if key=='file':
                        attr.extend(data[key]['data'])
                    else:
                        attr.extend(data[key]['data'])
                    setattr(self, key, attr)

            data = np.array([getattr(self, key) for key in columns]).transpose()
            [delattr(self, key) for key in columns]

            df = pd.DataFrame(data, columns=self.header)
            change = [key for key in columns if key != 'file']

            df = self._coerce_df_columns_to_numeric(df, change)
            return df

        else:

            index_name = list()
            new_data = list()
            for count, data in enumerate(self.doc):

                attrs = [data[key]['data'] for key in columns]
                index_name.append('Case{}'.format(count))
                new_data.append(pd.DataFrame(np.array(attrs).transpose(), columns=columns))
            data = pd.Series(new_data)
            data.index = index_name
            return data

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

            return self._to_dataframe()

        elif data_type.lower() == 'series':

            return self._to_series()

        elif data_type.lower() == 'csv':

            self._to_csv(**options)
            return self.doc

        else:

            raise NameError('{} is not possible convert to {}'.format(type(self).__name__, data_type.lower()))