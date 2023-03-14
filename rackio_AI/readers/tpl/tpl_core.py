import os
import re
import platform
import numpy as np
import pandas as pd
from easy_deco import progress_bar, raise_error
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
        # self.genkey = list()
        self.settings = dict()
        TPL._instances.append(self)

    def read(self, name, **kwargs):
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
        self.doc = self.__read_files(name, **kwargs)

        return self.doc

    def __read_file(self, filename, **kwargs):
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

        self.header = pd.MultiIndex.from_tuples(
            multi_index, names=['tag', 'variable', 'unit'])

        # Read Genkey

        if hasattr(self, '_join_files'):

            _attr = getattr(self, '_join_files')

            if not _attr:

                genkey_filename = filename.split(os.path.sep)
                genkey_filename.pop(-2)
                genkey_filename = os.path.join(*genkey_filename) + '.genkey'
                genkey = Genkey()
                genkey.read(filename=genkey_filename)
                doc['genkey'] = genkey

                # Provisional meanwhile read settings is not implemented.
                doc['settings'] = self.settings

        return doc

    @progress_bar(desc='Loading .tpl files...', unit='files', gen=True)
    def __read_files(self, filenames, **kwargs):
        """
        Read all .tpl files in a list of filenames

        ___
        **Parameters**

        **:param filenames:** list['str'] filenames list

        **:return:**

        * **doc:** (list[dict]) tpl file reformated in dictionaries

        """

        return self.__read_file(filenames, **kwargs)

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

        sections = file.split("{} \n".format(
            self.tpl_options.split_expression))

        self.tpl_options.header_line_numbers = int(sections[1].split('\n')[0])

        data_header_section = sections[1].split('\n')

        data = self.__get_data(
            data_header_section[self.tpl_options.header_line_numbers + 2::])

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
        header_section = data_header_section[1:
                                             self.tpl_options.header_line_numbers + 2]

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

    def __to_dataframe(self, join_files: bool = True):
        """
        ...Documentation here...

        **Parameters**

        None

        **:return:**

        """
        return self.__join(flag=join_files)

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
            # columns = self.doc[0].keys()
            index_name = list()
            new_data = list()

            for count, data in enumerate(self.doc):

                # print(f"data: {data}")
                columns = data.keys()
                attrs = [data[key] for key in columns if key !=
                         'genkey' and key != 'settings']
                # breakpoint()
                index_name.append('Case{}'.format(count))

                new_data.append({
                    'tpl': pd.DataFrame(np.array(attrs).transpose(), columns=self.header),
                    'genkey': data['genkey'],
                    'settings': data['settings']
                }
                )
            # breakpoint()
            # print(f"New Data: {new_data}")
            # data = pd.Series(new_data)
            # data.index = index_name

            return new_data

    def to(self, data_type, join_files: bool = True, **kwargs):
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
        options = {key: kwargs[key] if key in kwargs.keys(
        ) else kwargs_default[key] for key in kwargs_default.keys()}

        if data_type.lower() == 'dataframe':

            return self.__to_dataframe(join_files=join_files)

        elif data_type.lower() == 'series':

            return self.__to_series()

        elif data_type.lower() == 'csv':

            self.__to_csv(**options)

            return self.doc

        else:

            raise NameError('{} is not possible convert to {}'.format(
                type(self).__name__, data_type.lower()))

    def __making_dataframes(self, doc):
        """

        """
        for data in doc:

            yield pd.DataFrame(map(list, zip(*list(data.values()))), columns=self.header)


class Genkey(dict):

    def __init__(self, *args, **kwargs):
        self.__previous_line = None
        self.__previous_item = None
        self._keys = list()
        super().__init__(*args, **kwargs)

    def set_previous_item(self, item: str):

        self.__previous_item = item

    def get_previous_item(self) -> str:

        return self.__previous_item

    def set_previous_line(self, line: str):

        self.__previous_line = line

    def get_previous_line(self):

        return self.__previous_line

    def __append_key(self, key: str):

        if key not in self.__get_keys():

            self._keys.append(key)

    def __clean_keys(self):

        self._keys = list()

    def __clean_last_key(self):

        self._keys.pop(-1)

    def __get_keys(self):

        return self._keys

    # def __setitem__(self, key: str, value=None):

    #     if key in self.keys():

    #         _value = self.__getitem__(key)

    #         if _value is None:

    #             _value = list()

    #         if isinstance(_value, list):

    #             _value.append(value)

    #         return super().__setitem__(key, _value)

    #     return super().__setitem__(key, Genkey())

    # def __getitem__(self, key: str):

    #     return super().__getitem__(key)

    # def read(self, filename: str):
    #     r"""
    #     Documentation here"""
    #     with open(filename, "r") as file:

    #         lines = file.readlines()

    #     for line in lines:

    #         previous_line = self.get_previous_line()

    #         if previous_line:

    #             line = previous_line.replace("\\", line.lstrip())

    #         # Join line continuation
    #         if "\\" in line:
    #             line.replace("\\", "")
    #             self.set_previous_line(line)
    #             continue
    #         else:

    #             self.set_previous_line(None)

    #         if line.startswith("!*"):

    #             continue

    #         # if line.startswith(" "):

    #         #     continue

    #         # Setting first level keys
    #         if line.strip().startswith("! "):
    #             self.__clean_keys()
    #             key = line.strip().split("!")[1].strip()
    #             self.__append_key(key)
    #             self.__setitem__(key)

    #             # breakpoint()

    #             continue

    #         # Setting second level keys
    #         value = re.search('\w+\s', line)
    #         if not value:

    #             continue

    #         _key = value.group(0)
    #         _items = line.split(f"{_key}")[-1]
    #         self.__append_key(_key.lstrip().rstrip())

    #         _items = _items.split(", ")
    #         # print(f"Items: {_items}")

    #         # Iteration in last key - value
    #         for item in _items:
    #             # Is an item key - value
    #             breakpoint()
    #             if "=" in item:
    #                 previous_item = self.get_previous_item()
    #                 if previous_item:
    #                     key, value = self.get_previous_item().split("=")
    #                     self.__append_key(key)
    #                     continue
    #                     # print(f"Keys: {self.__get_keys()}")
    #                     # print(f"value: {value}")
    #                 self.set_previous_item(item)
    #                 # key, value = self.get_previous_item().split("=")
    #                 self.__setitem__(key=self._key)

    #                 self.__clean_last_key()
    #                 continue

    #             # This item belongs to previous item
    #             else:

    #                 _item = self.get_previous_item() + ", " + item
    #                 self.set_previous_item(_item)

    #         self.__clean_last_key()

    # TODO: Refactor and document the class' methods.

    def clean_lines(self, lines: str):
        '''
        Documentation here
        '''
        # Append lines when it has \\
        _el = ''
        broken_lines = []
        for el in lines.split('\n'):
            if re.search('\\\\', el):
                if not _el:
                    _el = el
                    continue
                _el += el
                continue

            if el.find('\\\\') == -1 and _el and bool(el.strip()):
                _el += el
                _el = ' '.join([e.strip() for e in _el.split('\\')])
                broken_lines.append(_el.strip())
                _el = ''
                continue

            if bool(el.strip()):
                broken_lines.append(el.strip())

        # Append lines when it starts with third level key
        _el = ''
        fixed_lines = []
        second_key_pattern = re.compile(r'^[a-zA-Z]+\s\w+')
        third_key_pattern = re.compile(r'^[a-zA-Z]+\=|^[a-zA-Z]+\s\=')

        for line in broken_lines:
            if second_key_pattern.search(line):
                _el = line
                fixed_lines.append(line)
                continue

            if third_key_pattern.search(line):
                line = ' ' + line
                _el += line
                fixed_lines.append(_el)
                continue

        return fixed_lines

    def split_values(self, line):
        '''
        Documentation here
        '''
        _info = ''
        _el = ''
        clean_line = []
        flag = False
        second_key_pattern = re.compile(r'^[A-Z]+\s')
        opening_third_key_pattern_1 = re.compile(
            r'^[A-Z]+\=\(|^[A-Z]+\s\=\s\(')
        opening_third_key_pattern_2 = re.compile(r'^[A-Z]+\=\(')
        third_key_pattern = re.compile(r'^[A-Z]+\=')
        closing_third_key_pattern = re.compile(r'\)$|\)\s.+$')

        for el in line.split(', '):
            if second_key_pattern.search(el):
                splited_line = el.split(' ')
                clean_line.append(splited_line[0])
                second_key = ' '.join([e for e in splited_line[1:]])

                if opening_third_key_pattern_1.search(second_key):
                    _el = second_key
                    flag = True
                    continue
                el = second_key
                clean_line.append(el)
                continue

            if opening_third_key_pattern_2.search(el):
                _el = el
                flag = True
                continue

            if third_key_pattern.search(el):
                if re.search(r'^INFO', el):
                    _info = el
                    continue

                clean_line.append(el)
                continue

            if re.search(r'^INFO', _info):
                _info = _info + ', ' + el
                clean_line.append(_info)
                _info = ''
                continue

            if flag:
                el = ', ' + el
                _el += el

                if closing_third_key_pattern.search(el):
                    clean_line.append(_el)
                    _el = ''
                    flag = False

        return clean_line

    def get_dict_values(self, values: list):
        '''
        Documentation here
        '''
        k = [el.split('=')[0].strip() for el in values]
        v = [el.split('=')[1].strip() for el in values]
        k_v = dict(zip(k, v))

        pattern = re.compile(r'\d\s\w|\d\)\s\w+|\d\)\s\%|\d\s\%|\(\"\w+')
        for key, val in k_v.items():
            if re.search(r'\(\"\.\./|\(\"\w+', val):
                val = [e.replace('"', '').replace('(', '').replace(')', '').strip()
                       for e in val.split(',')]
                val = tuple(val)
                k_v[key] = val
                continue

            if re.search(r'^INFO', key):
                val = val.replace('"', '')
                k_v[key] = val
                continue

            if re.search(r'PVTFILE', key) and not re.search(r'\(\"\.\./|\(\"\w+', val):
                k_v[key] = val.replace('"', '')
                continue

            if pattern.search(val):
                if re.search(r'TERMINALS', key):
                    val = val.replace('(', '').replace(')',
                                                       '').replace(',', '')
                    val = [e.strip() for e in val.split(' ')]
                    _val = []
                    _el = ''
                    n = 0
                    for el in val:
                        n += 1
                        if n == 1:
                            _el = el
                            continue

                        if n == 2:
                            el = ' ' + el
                            _el += el
                            _val.append(_el)
                            n = 0
                            continue
                    VALUE = tuple(_val)
                    k_v[key] = VALUE
                    continue
                else:
                    val = val.split(' ')
                    VALUE = ' '.join([el for el in val[:-1]])
                    UNIT = val[-1]
                    plural = False
                    VALUE = eval(VALUE)

                    if isinstance(VALUE, tuple):
                        plural = True

                    k_v[key] = {
                        f'VALUE{"S" if plural else ""}': VALUE,
                        'UNIT': UNIT.strip(',')
                    }
                    continue

            if re.search(r'\d+\.\d+|^[0-9]*$', val):
                k_v[key] = eval(val)
                continue

            k_v[key] = val.replace('"', '')

        return k_v

    def read(self, filename: str):
        '''
        Documentation here
        '''
        assert isinstance(
            filename, str), f'filename must be a string! Not {type(filename)}'
        
        try:

            with open(filename, 'r') as f:
                file = f.read()

        except:

            with open(os.path.sep + os.path.join(filename), 'r') as f:
                file = f.read()

        # Splitting Genkey in principal elements
        split_genkey_elements_pattern = re.compile('\s\n')
        genkey_elements = []

        for element in split_genkey_elements_pattern.split(file):
            genkey_elements.append(element)

        # Getting first level and second level Genkey keys
        first_level_key_pattern = re.compile('!\s\w+.+')
        first_level_keys = []
        second_level_keys = []

        for el in genkey_elements:
            genkey_element = ' '.join([c.strip() for c in el.split(' ')])
            _first_level_key = first_level_key_pattern.search(genkey_element)

            if _first_level_key:
                first_level_key = _first_level_key.group().replace('!', '').strip()
                first_level_keys.append(first_level_key)

                lines = self.clean_lines(el)
                elements = list(map(self.split_values, lines))
                second_keys = [el[0] for el in elements]
                list_values = [el[1:] for el in elements]
                values = list(map(self.get_dict_values, list_values))
                key_vals_list = list(zip(second_keys, values))

                key_vals_dict = {}
                for key in key_vals_list:
                    key_vals_dict.setdefault(key[0], []).append(key[1])

                for key, val in key_vals_dict.items():
                    if len(val) == 1:
                        key_vals_dict[key] = key_vals_dict.get(key)[0]

                second_level_keys.append(key_vals_dict)

        # Putting together first and second level keys
        genkey_keys = list(zip(first_level_keys, second_level_keys))

        # Creating list of second level keys for duplicated first level keys
        for key in genkey_keys:
            self.setdefault(key[0], []).append(key[1])

        # Extracting second level keys from list if first level key is not duplicated.
        for key, val in self.items():
            if len(val) == 1:
                self[key] = self.get(key)[0]

        for key, val in self.items():
            if val == {}:
                self[key] = None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
