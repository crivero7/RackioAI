import os
import glob
from rackio_AI.readers.tpl import TPL


class Reader:
    """
        In all data analysis projects you must load data from different file extensions, so, the **Reader** class has that
        software intention, it read different file extensions and convert them into a *pandas.DataFrame* or *np.ndarray* to add
        them to your project structure in **RackioAI**

        So far, you only can *.tpl* files. This file extensions are proper of [OLGA](https://www.software.slb.com/products/olga)
        Dynamic Multiphase Flow Simulator

    """
    tpl = TPL()

    def __init__(self):
        """

        """
        pass

    def read(self, filename):
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
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> cwd = os.getcwd()

        ```
        ## An especific file

        ```python
        >>> filename = os.path.join('..','data', 'Leak')
        >>> RackioAI.load(filename)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak111
        1            0.502732  ...  Leak111
        2            1.232772  ...  Leak111
        3            1.653696  ...  Leak111
        4            2.200430  ...  Leak111
        ...               ...  ...      ...
        32182     1618.124000  ...  Leak120
        32183     1618.662000  ...  Leak120
        32184     1619.200000  ...  Leak120
        32185     1619.737000  ...  Leak120
        32186     1620.275000  ...  Leak120
        <BLANKLINE>
        [32187 rows x 12 columns]

        ```

        ## A directory

        ```python
        >>> directory = os.path.join('..','data', 'Leak')
        >>> RackioAI.load(directory)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak111
        1            0.502732  ...  Leak111
        2            1.232772  ...  Leak111
        3            1.653696  ...  Leak111
        4            2.200430  ...  Leak111
        ...               ...  ...      ...
        64369     1618.124000  ...  Leak120
        64370     1618.662000  ...  Leak120
        64371     1619.200000  ...  Leak120
        64372     1619.737000  ...  Leak120
        64373     1620.275000  ...  Leak120
        <BLANKLINE>
        [64374 rows x 12 columns]

        ```
        """
        (_, file_extension) = os.path.splitext(filename)
        tpl_file =False

        if file_extension=='.tpl':

            tpl_file = True

        elif self.check_extension_files(filename, ext='.tpl'):

            tpl_file = True

        if tpl_file:

            self.tpl.read(filename)
            data = self.tpl.to('dataframe')

            return data

        return None

    @staticmethod
    def check_extension_files(root_directory, ext='.tpl'):
        """
        This is an utility method which you can check if in any directory exist files with *:param ext* extension

        ___
        **Parameters**

        * **:param root_directory:** (str) directory to look for files
        * **:param ext** (str) default='.tpl' extension file to look for

        **:return:**

        * **bool**: If True, exist *ext* in root_directory}

        ___

        ## Snippet code

        ```python
        >>> directory = os.path.join('..','data', 'Leak')
        >>> Reader.check_extension_files(directory)
        True

        ```
        """

        files = [f for f in glob.glob(root_directory + "**/*{}".format(ext), recursive=True)]

        if files:

            return True

        else:

            return False

if __name__=="__main__":
    import doctest
    doctest.testmod()