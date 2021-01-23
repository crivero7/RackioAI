import os
import numpy as np
import pandas as pd
from easy_deco import progress_bar, raise_error
from rackio_AI.utils import Utils
from rackio_AI.readers.tpl.options import TPLOptions
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


@set_to_methods(del_temp_attr)
class TPL:
    """
    **TPL** class allows to you load into RackioAI .tpl files in pandas.DataFrame format.
    """

    tpl_options = TPLOptions()
    _instances = list()       

    def __init__(self):

        self.file_extension = ".tpl"
        self.doc = list()
        TPL._instances.append(self)

    def read(self, name):
        """
        Read .tpl files

        ___
        **Paramaters**

        * **:param name:** (str) if *name* is a directory, it reads all .tpl files in that directory.
        If *name* is a .tpl file, it reads only that file

        **:return:**

        * **doc:** (list[dict]) tpl file reaformated in dictionaries
        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
        >>> name = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.load(name)
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        3214      1618.327000                                   569341.1  ...                           0.0   Leak01
        3215      1618.849000                                   569341.3  ...                           0.0   Leak01
        3216      1619.370000                                   569341.5  ...                           0.0   Leak01
        3217      1619.892000                                   569341.7  ...                           0.0   Leak01
        3218      1620.413000                                   569341.9  ...                           0.0   Leak01
        <BLANKLINE>
        [3219 rows x 12 columns]

        ```
        """
        self.doc = self.__read_files(name)
        
        return self.doc

    def __read_file(self, filename):
        """
        Read only one .tpl file

        ___
        **Parameters**

        **:param filename:** (str) tpl filename

        **:return:**

        * **doc:** (dict) .tpl file in a dictionary

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
        >>> name = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.load(name)
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        3214      1618.327000                                   569341.1  ...                           0.0   Leak01
        3215      1618.849000                                   569341.3  ...                           0.0   Leak01
        3216      1619.370000                                   569341.5  ...                           0.0   Leak01
        3217      1619.892000                                   569341.7  ...                           0.0   Leak01
        3218      1620.413000                                   569341.9  ...                           0.0   Leak01
        <BLANKLINE>
        [3219 rows x 12 columns]

        ```
        """

        doc = dict()

        (data_header_section, data) = self.__get_section_from(filename)
        header_section = self.__get_header_section(data_header_section)
        (filename, _) = os.path.splitext(filename)

        multi_index = list()

        for count, column_name in enumerate(header_section):
            column_name = self.__clean_column_name(column_name)

            (tag, unit, variable_type) = self.__get_structure(column_name)
            multi_index.append((tag, variable_type, unit))
            "fill dictionary"
            doc[tag] = data[:, count]

        data_name = np.array([filename.split(os.path.sep)[-1]] * data.shape[0])

        multi_index.append(('file', 'filename', '.tpl'))
        doc['file'] = data_name

        self.header = pd.MultiIndex.from_tuples(multi_index, names=['tag', 'variable', 'unit'])

        return doc

    @progress_bar(desc='Loading .tpl files...', unit='files', gen=True)
    def __read_files(self, filenames):
        """
        Read all .tpl files in a list of filenames

        ___
        **Parameters**

        **:param filenames:** list['str'] filenames list

        **:return:**

        * **doc:** (list[dict]) tpl file reformated in dictionaries

        """

        return self.__read_file(filenames)

    @raise_error
    def __get_section_from(self, filename):
        """
        Get time profile section separated by key word  in tpl_options.split_expression, for OLGA .tpl files this key is
        CATALOG

        ___
        **Parameters**

        * **filename:** (str)

        **:return:**

        * **(data_header_section, data):**
            * **data_header_section:** list['str'] .tpl file header section
            * **data:** (np.ndarray) data section

        ```
        """
        with open(filename, 'r') as file:
            file = file.read()

        sections = file.split("{} \n".format(self.tpl_options.split_expression))

        self.tpl_options.header_line_numbers = int(sections[1].split('\n')[0])

        data_header_section = sections[1].split('\n')

        data = self.__get_data(data_header_section[self.tpl_options.header_line_numbers + 2::])

        return data_header_section, data

    def __get_header_section(self, data_header_section):
        """
        Get header section tag description of .tpl file

        ___
        **Parameters**

        * **data_header_section:** list['str'] .tpl file header section

        **:return:**

        * **header_section:** list('str') each item in the list is tag variable summary in .tpl files

        """
        header_section = data_header_section[1: self.tpl_options.header_line_numbers + 2]

        return header_section[-1:] + header_section[:-1]

    @staticmethod
    def __get_data(data):
        """
        Get time profile section separated by key word  in tpl_options.split_expression, for OLGA .tpl files this key is
        CATALOG

        **Parameters**

        * **data:** (np.ndarray) data section with elements in np.ndarray are strings
        **:return:**

        * **data:** (np.ndarray)
        """
        rows = len(data)
        new_data = list()

        for count, d in enumerate(data):

            if count == rows - 1:
                break

            new_data.append(np.array([float(item) for item in d.split(" ")]))

        return np.array(new_data)

    @staticmethod
    def __clean_column_name(column_name):
        """

        **Parameters**

        * **:param column_name:** (str)

        **:return:**

        * **column_name:** ('str')

        """

        return column_name.replace("'", "").replace(":", "").replace(" ", "_").replace("-", "").replace("__", "_")

    @staticmethod
    def __get_tag(column_name):
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

    @staticmethod
    def __get_unit(column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        return column_name[column_name.find("(") + 1:column_name.find(")")]

    @staticmethod
    def __get_variable_type(column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        return column_name[column_name.find(")") + 2::]

    def __get_structure(self, column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        tag = self.__get_tag(column_name)

        unit = self.__get_unit(column_name)

        variable_type = self.__get_variable_type(column_name)

        return tag, unit, variable_type

    def __to_dataframe(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return self.__join(flag=True)

    def __to_series(self):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return self.__join(flag=False)

    def __to_csv(self, **kwargs):
        """
        ...Documentation here...

        **Paramters**

        * **:param kwargs:**

        **:return:**

        """
        df = self.__join(flag=True)

        df.to_csv(os.path.join(kwargs['path'], kwargs['filename']))

    @staticmethod
    def __coerce_df_columns_to_numeric(df, column_list):
        """
        ...Documentation here...

        **Parameters**

        * **:param df:**
        * **:param column_list:**

        **:return:**

        """
        df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')

        return df

    def __join(self, flag=True):
        """
        ...Documentation here...

        **Parameters**

        * **:param flag:**

        **:return:**

        """
        if flag:

            # Making dataframes
            d = self.__making_dataframes(self.doc)
            df = pd.concat(d)
            change = [key[0] for key in self.header.values if key[0] != 'file']
            df = self.__coerce_df_columns_to_numeric(df, change)

            return df

        else:
            columns = self.doc[0].keys()
            index_name = list()
            new_data = list()

            for count, data in enumerate(self.doc):
                attrs = [data[key]['data'] for key in columns]
                index_name.append('Case{}'.format(count))
                new_data.append(pd.DataFrame(np.array(attrs).transpose(), columns=self.header))

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

            return self.__to_dataframe()

        elif data_type.lower() == 'series':

            return self.__to_series()

        elif data_type.lower() == 'csv':

            self.__to_csv(**options)

            return self.doc

        else:

            raise NameError('{} is not possible convert to {}'.format(type(self).__name__, data_type.lower()))

    def __making_dataframes(self, doc):
        """

        """
        for data in doc:

            yield pd.DataFrame(map(list, zip(*list(data.values()))), columns=self.header)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
