import os
import numpy as np
import pandas as pd
from rackio_AI.decorators.progress_bar import progressBar
from .options import TPLOptions


class TPL:
    """
    ...Documentation here...
    """

    tpl_options = TPLOptions()

    def __init__(self):
        """
        ...Documentation here...
        """
        self.file_extension = ".tpl"
        self.doc = list()


    def read(self, filename, specific_file=True):
        """
        ...Documentation here...

        **Paramaters**

        * **:param filename:**
        * **:param specific_file:**

        **:return:**

        """
        if specific_file:

            self.doc = [self._read_file(filename)]

        else:

            self.doc = self._read_all_files(filename)

        return self.doc

    def _read_file(self, filename):
        """
        ...Documentation here...

        **Parameters**

        **:param filename:**

        **:return:**

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
        ...Documentation here...

        **Parameters**

        * **:param directory:**

        **:return:**

        """
        filenames = self.find_files(self.tpl_options.file_extension, directory)

        doc = self._read_files(filenames)

        self.doc = doc

        return doc

    @progressBar(desc='Loading files...', unit='files')
    def _read_files(self, filenames):
        """
        ...Documentation here...

        **Parameters**

        * **:param filenames:**

        **:return:**

        """

        self.doc.append(self._read_file(filenames))

        return self.doc

    def _get_section_from(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        sections = self.file.split("{} \n".format(self.tpl_options.split_expression))

        self.tpl_options.header_line_numbers = int(sections[1].split('\n')[0])

        data_header_section = sections[1].split('\n')

        data= self._get_data(data_header_section[self.tpl_options.header_line_numbers + 2::])

        return (data_header_section, data)

    def _get_header_section(self, data_header_section):
        """
        ...Documentation here...

        **Parameters**

        * **:param data_header_section:**

        **:return:**

        """
        header_section = data_header_section[1: self.tpl_options.header_line_numbers + 2]

        return header_section[-1:] + header_section[:-1]

    def _get_data(self, data):
        """
        ...Documentation here...

        **Parameters**

        * **:param data:**

        **:return:**

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
        ...Documentation here...

        **Parameters**

        * **:param extension:**
        * **:param path:**

        **:return:**

        """
        result = list()

        for root, dirs, files in os.walk(path):

            for file in files:

                if file.endswith(extension):

                    result.append(os.path.join(root, file))

        return result

    def _clean_column_name(self, column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """

        return column_name.replace("'", "").replace(":", "").replace(" ", "_").replace("-", "").replace("__","_")

    def _get_tag(self, column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        tag = column_name[0:column_name.find("(") - 1]

        if tag.endswith("_"):

            tag = tag[0:-1]

        return tag

    def _get_unit(self, column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        return column_name[column_name.find("(") + 1:column_name.find(")")]

    def _get_variable_type(self, column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        return column_name[column_name.find(")")+2::]

    def _get_structure(self, column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        tag = self._get_tag(column_name)

        unit = self._get_unit(column_name)

        variable_type = self._get_variable_type(column_name)

        return (tag, unit, variable_type)

    def _to_dataframe(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return self._join(flag=True)

    def _to_series(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return self._join(flag=False)

    def _to_csv(self, **kwargs):
        """
        ...Documentation here...

        **Paramters**

        * **:param kwargs:**

        **:return:**

        """
        df = self._join(flag=True)

        df.to_csv(os.path.join(kwargs['path'], kwargs['filename']))

    @staticmethod
    def _coerce_df_columns_to_numeric(df, column_list):
        """
        ...Documentation here...

        **Parameters**

        * **:param df:**
        * **:param column_list:**

        **:return:**

        """
        df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')

        return df

    def _join(self, flag=True):
        """
        ...Documentation here...

        **Parameters**

        * **:param flag:**

        **:return:**

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
        This method allows to you transform from .tpl to a 'data_type'

        **Parameers**

        * **:param data_type:** (str) 'dataframe' - 'series' - 'csv'
        * **:param kwargs:**
            * **filename:** (str) 'name.csv' if date_type == 'csv'
            * **path:** (str ) path to save csv file

        **:return:**

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