import os
from rackio_AI.readers.tpl import TPL
from rackio_AI.readers._csv_.core import CSV
from rackio_AI.utils import Utils


class Reader:
    """
        In all data analysis projects you must load data from different file extensions, so, the **Reader** class has that
        software intention, it read different file extensions and convert them into a *pandas.DataFrame* or *np.ndarray* to add
        them to your project structure in **RackioAI**

        So far, you only can *.tpl* files. This file extensions are proper of [OLGA](https://www.software.slb.com/products/olga)
        Dynamic Multiphase Flow Simulator

    """
    tpl = TPL()
    _csv = CSV()

    def __init__(self):
        """

        """
        pass

    def read(self, filename, file_type=".tpl"):
        """
        Read data supported by RackioAI in pandas.DataFrame

        ___
        **Parameters**

        * **:param filename:** (str) Can be a directory or a filename with its extension

        **:return:**

        * **data:** (pandas.DataFrame)

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory

        ```
        ## An especific file

        ```python
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.load(filename)
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

        ## A directory

        ```python
        >>> directory = os.path.join(get_directory('Leak'))
        >>> RackioAI.load(directory)
        tag       TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                                Pressure  ...             Controller_output filename
        unit                S                                         PA  ...                                   .tpl
        0            0.000000                                   568097.3  ...                           0.0   Leak01
        1            0.502732                                   568098.2  ...                           0.0   Leak01
        2            1.232772                                   568783.2  ...                           0.0   Leak01
        3            1.653696                                   569367.3  ...                           0.0   Leak01
        4            2.200430                                   569933.5  ...                           0.0   Leak01
        ...               ...                                        ...  ...                           ...      ...
        38616     1618.124000                                   569345.4  ...                           0.0  Leak120
        38617     1618.662000                                   569345.6  ...                           0.0  Leak120
        38618     1619.200000                                   569345.7  ...                           0.0  Leak120
        38619     1619.737000                                   569345.8  ...                           0.0  Leak120
        38620     1620.275000                                   569346.0  ...                           0.0  Leak120
        <BLANKLINE>
        [38621 rows x 12 columns]

        ```
        """
        (_, file_extension) = os.path.splitext(filename)
        tpl_file = False
        filenames = Utils.check_extension_files(filename, ext=file_type)

        if filenames:
            try:
                self.tpl.read(filename)
                data = self.tpl.to('dataframe')

            except:
                raise FileNotFoundError('{} is not found'.format(filename))

            return data

        if file_extension == '.tpl':

            tpl_file = True

        elif filenames:

            tpl_file = True
            filename = filenames

        if tpl_file:
            try:
                self.tpl.read(filename)
                data = self.tpl.to('dataframe')

            except:
                raise FileNotFoundError('{} is not found'.format(filename))

            return data


    
if __name__ == "__main__":
    import doctest

    doctest.testmod()
