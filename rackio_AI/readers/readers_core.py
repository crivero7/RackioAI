import os
from rackio_AI.readers.tpl import TPL
from rackio_AI.readers._csv_.csv_core import CSV
from rackio_AI.readers.pkl.pkl_core import PKL
from rackio_AI.readers.exl.exl_core import EXL


class Reader:
    """
    In all data analysis projects you must load data from different file extensions, so, the **Reader** class has that
    software intention, it read different file extensions and convert them into a *pandas.DataFrame* to add them to your 
    project structure in **RackioAI**

    You can read *.tpl*, *.csv*, *excel* files and pickled object as DataFrame.
    .tpl files extension are proper of [OLGA](https://www.software.slb.com/products/olga) 
    Dynamic Multiphase Flow Simulator

    """
    tpl = TPL()
    _csv = CSV()
    pkl = PKL()
    exl = EXL()

    def read(self, pathname: str, ext: str=".tpl", **kwargs):
        """
        Read data supported by RackioAI in pandas.DataFrame

        ___
        **Parameters**

        * **:param pathname:** (str) Can be a directory or a filename

        **:return:**

        * **data:** (pandas.DataFrame)

        ___

        ## Snippet code

        ### **Olga TPL files**

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
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
        6429      1617.966000                                   569342.5  ...                           0.0   Leak02
        6430      1618.495000                                   569342.8  ...                           0.0   Leak02
        6431      1619.025000                                   569343.0  ...                           0.0   Leak02
        6432      1619.554000                                   569343.2  ...                           0.0   Leak02
        6433      1620.083000                                   569343.4  ...                           0.0   Leak02
        <BLANKLINE>
        [6434 rows x 12 columns]

        ```

        ### **CSV files**

        ```python
        >>> directory = os.path.join(get_directory('csv'), "standard")
        >>> RackioAI.load(directory, ext=".csv", delimiter=";", header=0)
            Username   Identifier One-time password Recovery code First name Last name   Department    Location
        0   booker12         9012            12se74        rb9012     Rachel    Booker        Sales  Manchester
        1     grey07         2070            04ap67        lg2070      Laura      Grey        Depot      London
        2  johnson81         4081            30no86        cj4081      Craig   Johnson        Depot      London
        3  jenkins46         9346            14ju73        mj9346       Mary   Jenkins  Engineering  Manchester
        4    smith79         5079            09ja61        js5079      Jamie     Smith  Engineering  Manchester
        
        ```
        """
        if ext==".tpl":

            self.tpl.read(pathname)
            data = self.tpl.to('dataframe')

        elif ext==".csv":

            data = self._csv.read(pathname, **kwargs)

        elif ext==".pkl":
            
            data = self.pkl.read(pathname, **kwargs)

            return data

        elif ext in [".xls", ".xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:

            data = self.exl.read(pathname, **kwargs)

            return data

        else:

            raise TypeError("File format not supported")

        return data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
