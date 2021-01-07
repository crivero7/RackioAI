import os
import sys
import numpy as np
import pandas as pd
from easy_deco import progress_bar, raise_error
from rackio_AI.readers.options import TPLOptions


class TPL:
    """
    **TPL** class allows to you load into RackioAI .tpl files in pandas.DataFrame format.
    """

    tpl_options = TPLOptions()

    def __init__(self):

        self.file_extension = ".tpl"
        self.doc = list()

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
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI.data import get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## An especific file
        >>> name = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.reader.tpl.read(name)
        [{'TIME_SERIES': {'variable': '', 'unit': 'S', 'data': array([0.000000e+00, 5.027318e-01, 1.232772e+00, ..., 1.619370e+03,
               1.619892e+03, 1.620413e+03])}, 'PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1': {'variable': 'Pressure', 'unit': 'PA', 'data': array([568097.3, 568098.2, 568783.2, ..., 569341.5, 569341.7, 569341.9])}, 'TM_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1': {'variable': 'Fluid_temperature', 'unit': 'C', 'data': array([-41.76985, -41.76985, -41.76967, ..., -41.2957 , -41.29513,
               -41.29456])}, 'GT_BOUNDARY_BRANCH_TUBERIA_PIPE_Pipe60_NR_1': {'variable': 'Total_mass_flow', 'unit': 'KG/S', 'data': array([37.83052, 37.83918, 37.83237, ..., 36.98472, 36.98392, 36.98313])}, 'PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe151_NR_1': {'variable': 'Pressure', 'unit': 'PA', 'data': array([352683.3, 353449.8, 353587.3, ..., 353602. , 353602.1, 353602.1])}, 'TM_SECTION_BRANCH_TUBERIA_PIPE_Pipe151_NR_1': {'variable': 'Fluid_temperature', 'unit': 'C', 'data': array([-41.81919, -41.81898, -41.81895, ..., -40.90041, -40.9004 ,
               -40.90039])}, 'GT_BOUNDARY_BRANCH_TUBERIA_PIPE_Pipe151_NR_1': {'variable': 'Total_mass_flow', 'unit': 'KG/S', 'data': array([37.83052, 37.70243, 37.67011, ..., 36.96286, 36.96208, 36.96132])}, 'PUMPSPEED_PUMP_PUMP': {'variable': 'Pump_speed', 'unit': 'RPM', 'data': array([1300., 1300., 1300., ..., 1300., 1300., 1300.])}, 'GTLEAK_LEAK_FUGA': {'variable': 'Leakage_total_mass_flow_rate', 'unit': 'KG/S', 'data': array([0., 0., 0., ..., 0., 0., 0.])}, 'CONTR_CONTROLLER_CONTROL_BOMBA': {'variable': 'Controller_output', 'unit': '', 'data': array([0., 0., 0., ..., 0., 0., 0.])}, 'CONTR_CONTROLLER_CONTROL_FUGA': {'variable': 'Controller_output', 'unit': '', 'data': array([0., 0., 0., ..., 0., 0., 0.])}, 'file': {'variable': 'file', 'unit': '.tpl', 'data': array(['Leak01', 'Leak01', 'Leak01', ..., 'Leak01', 'Leak01', 'Leak01'],
              dtype='<U6')}}]

        ```
        """
        if os.path.isfile(name):

            self.doc = [self._read_file(name)]

            return self.doc

        elif os.path.isdir(name):

            self.doc = self._read_all_files(name)

            return self.doc

        return None

    def _read_file(self, filename):
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
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI import get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## An especific file
        >>> name = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.reader.tpl._read_file(name)
        {'TIME_SERIES': {'variable': '', 'unit': 'S', 'data': array([0.000000e+00, 5.027318e-01, 1.232772e+00, ..., 1.619370e+03,
               1.619892e+03, 1.620413e+03])}, 'PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1': {'variable': 'Pressure', 'unit': 'PA', 'data': array([568097.3, 568098.2, 568783.2, ..., 569341.5, 569341.7, 569341.9])}, 'TM_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1': {'variable': 'Fluid_temperature', 'unit': 'C', 'data': array([-41.76985, -41.76985, -41.76967, ..., -41.2957 , -41.29513,
               -41.29456])}, 'GT_BOUNDARY_BRANCH_TUBERIA_PIPE_Pipe60_NR_1': {'variable': 'Total_mass_flow', 'unit': 'KG/S', 'data': array([37.83052, 37.83918, 37.83237, ..., 36.98472, 36.98392, 36.98313])}, 'PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe151_NR_1': {'variable': 'Pressure', 'unit': 'PA', 'data': array([352683.3, 353449.8, 353587.3, ..., 353602. , 353602.1, 353602.1])}, 'TM_SECTION_BRANCH_TUBERIA_PIPE_Pipe151_NR_1': {'variable': 'Fluid_temperature', 'unit': 'C', 'data': array([-41.81919, -41.81898, -41.81895, ..., -40.90041, -40.9004 ,
               -40.90039])}, 'GT_BOUNDARY_BRANCH_TUBERIA_PIPE_Pipe151_NR_1': {'variable': 'Total_mass_flow', 'unit': 'KG/S', 'data': array([37.83052, 37.70243, 37.67011, ..., 36.96286, 36.96208, 36.96132])}, 'PUMPSPEED_PUMP_PUMP': {'variable': 'Pump_speed', 'unit': 'RPM', 'data': array([1300., 1300., 1300., ..., 1300., 1300., 1300.])}, 'GTLEAK_LEAK_FUGA': {'variable': 'Leakage_total_mass_flow_rate', 'unit': 'KG/S', 'data': array([0., 0., 0., ..., 0., 0., 0.])}, 'CONTR_CONTROLLER_CONTROL_BOMBA': {'variable': 'Controller_output', 'unit': '', 'data': array([0., 0., 0., ..., 0., 0., 0.])}, 'CONTR_CONTROLLER_CONTROL_FUGA': {'variable': 'Controller_output', 'unit': '', 'data': array([0., 0., 0., ..., 0., 0., 0.])}, 'file': {'variable': 'file', 'unit': '.tpl', 'data': array(['Leak01', 'Leak01', 'Leak01', ..., 'Leak01', 'Leak01', 'Leak01'],
              dtype='<U6')}}

        ```
        """

        doc = dict()

        (data_header_section, data) = self._get_section_from(filename)
        header_section = self._get_header_section(data_header_section)
        (filename, _) = os.path.splitext(filename)

        multi_index = list()

        for count, column_name in enumerate(header_section):
            column_name = self._clean_column_name(column_name)

            (tag, unit, variable_type) = self._get_structure(column_name)
            multi_index.append((tag, variable_type, unit))
            "fill dictionary"
            doc[tag] = data[:, count]

        data_name = np.array([filename.split(os.path.sep)[-1]] * data.shape[0])

        multi_index.append(('file', 'filename', '.tpl'))
        doc['file'] = data_name

        self.header = pd.MultiIndex.from_tuples(multi_index, names=['tag', 'variable', 'unit'])

        return doc

    def _read_all_files(self, directory):
        """
        Read all .tpl files in a directory

        ___
        **Parameters**

        **:param directory:** (str) directory with .tpl files

        **:return:**

        * **doc:** (list[dict]) tpl file reformated in dictionaries

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI import get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## An especific file
        >>> directory = os.path.join(get_directory('Leak'))
        >>> data = RackioAI.reader.tpl._read_all_files(directory)

        ```
        """
        filenames = self.find_files(self.tpl_options.file_extension, directory)

        doc = self._read_files(filenames)

        self.doc = doc

        return doc

    @progress_bar(desc='Loading files...', unit='files')
    def _read_files(self, filenames, init=0):
        """
        Read all .tpl files in a list of filenames

        ___
        **Parameters**

        **:param filenames:** list['str'] filenames list

        **:return:**

        * **doc:** (list[dict]) tpl file reformated in dictionaries

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI import get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## A directory
        >>> directory = os.path.join(get_directory('Leak'))
        >>> filenames = RackioAI.reader.tpl.find_files(RackioAI.reader.tpl.tpl_options.file_extension, directory)
        >>> data = RackioAI.reader.tpl._read_files(filenames)

        ```
        """

        counter = init
        for filename in filenames:
            counter += 1
            print(counter)
            yield self._read_file(filename)

    @raise_error
    def _get_section_from(self, filename):
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

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI import get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## An especific file
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> data = RackioAI.load(filename)
        >>> (data_header_section, data) = RackioAI.reader.tpl._get_section_from(filename)

        ```
        """
        with open(filename, 'r') as file:
            file = file.read()

        sections = file.split("{} \n".format(self.tpl_options.split_expression))

        self.tpl_options.header_line_numbers = int(sections[1].split('\n')[0])

        data_header_section = sections[1].split('\n')

        data = self._get_data(data_header_section[self.tpl_options.header_line_numbers + 2::])

        return data_header_section, data

    def _get_header_section(self, data_header_section):
        """
        Get header section tag description of .tpl file

        ___
        **Parameters**

        * **data_header_section:** list['str'] .tpl file header section

        **:return:**

        * **header_section:** list('str') each item in the list is tag variable summary in .tpl files

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI import get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## An especific file
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> data = RackioAI.load(filename)
        >>> (data_header_section, data) = RackioAI.reader.tpl._get_section_from(filename)
        >>> RackioAI.reader.tpl._get_header_section(data_header_section)
        ["TIME SERIES  ' (S)  '", "PT 'SECTION:' 'BRANCH:' 'TUBERIA' 'PIPE:' 'Pipe-60' 'NR:' '1'  '(PA)' 'Pressure'", "TM 'SECTION:' 'BRANCH:' 'TUBERIA' 'PIPE:' 'Pipe-60' 'NR:' '1'  '(C)' 'Fluid temperature'", "GT 'BOUNDARY:' 'BRANCH:' 'TUBERIA' 'PIPE:' 'Pipe-60' 'NR:' '1'  '(KG/S)' 'Total mass flow'", "PT 'SECTION:' 'BRANCH:' 'TUBERIA' 'PIPE:' 'Pipe-151' 'NR:' '1'  '(PA)' 'Pressure'", "TM 'SECTION:' 'BRANCH:' 'TUBERIA' 'PIPE:' 'Pipe-151' 'NR:' '1'  '(C)' 'Fluid temperature'", "GT 'BOUNDARY:' 'BRANCH:' 'TUBERIA' 'PIPE:' 'Pipe-151' 'NR:' '1'  '(KG/S)' 'Total mass flow'", "PUMPSPEED 'PUMP:' 'PUMP' '(RPM)' 'Pump speed'", "GTLEAK 'LEAK:' 'FUGA' '(KG/S)' 'Leakage total mass flow rate'", "CONTR 'CONTROLLER:' 'CONTROL - BOMBA' '(-)' 'Controller output'", "CONTR 'CONTROLLER:' 'CONTROL - FUGA' '(-)' 'Controller output'"]

        ```
        """
        header_section = data_header_section[1: self.tpl_options.header_line_numbers + 2]

        return header_section[-1:] + header_section[:-1]

    @staticmethod
    def _get_data(data):
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
    def find_files(extension, path):
        """
        find all *:param extension:* files in *path*

        ___
        **Parameters**

        * **:param extension:** (str)
        * **:param path:** (str) root path

        **:return:**

        * **files:** (list['str'])

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI
        >>> from rackio_AI import get_directory
        >>> from rackio_AI.readers.tpl import TPL
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ## An especific file
        >>> path = os.path.join(get_directory('Leak'))
        >>> filenames = TPL.find_files('.tpl', path)

        ```
        """
        result = list()

        for root, dirs, files in os.walk(path):

            for file in files:

                if file.endswith(extension):
                    result.append(os.path.join(root, file))

        return result

    @staticmethod
    def _clean_column_name(column_name):
        """

        **Parameters**

        * **:param column_name:** (str)

        **:return:**

        * **column_name:** ('str')

        """

        return column_name.replace("'", "").replace(":", "").replace(" ", "_").replace("-", "").replace("__", "_")

    @staticmethod
    def _get_tag(column_name):
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
    def _get_unit(column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        return column_name[column_name.find("(") + 1:column_name.find(")")]

    @staticmethod
    def _get_variable_type(column_name):
        """
        ...Documentation here...

        **Parameters**

        * **:param column_name:**

        **:return:**

        """
        return column_name[column_name.find(")") + 2::]

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

        return tag, unit, variable_type

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

            # Making dataframes
            d = self.__making_dataframes(self.doc)
            df = pd.concat(d)
            change = [key for key in columns if key != 'file']
            df = self._coerce_df_columns_to_numeric(df, change)

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

    def __making_dataframes(self, doc):
        """

        """
        for data in doc:
            
            yield pd.DataFrame(map(list, zip(*list(data.values()))), columns=self.header)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
