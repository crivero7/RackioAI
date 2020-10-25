import os
import glob
from rackio_AI.readers.tpl import TPL


class Reader:
    """
        In all data analysis projects you must load data from different file extensions, so, the **Readers** class has that
        software intention, read different file extensions and convert them into a *pandas.DataFrame* or *np.ndarray* to add
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
        read data supported by RackioAI in pandas.DataFrame

        **Parameters**

        * **:param filename:** (str) Can be a directory or a filename with its extension

        **:return:**

        * **data:** (pandas.DataFrame)

        # Snippet code
        ```python
        >>> import os
        >>> from rackio_AI import RackioAI
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> cwd = os.getcwd()

        ## An especific file
        >>> filename = os.path.join(cwd, 'data', 'Leak','Leak112.tpl')
        >>> RackioAI.load(filename)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak112
        1            0.502732  ...  Leak112
        2            1.232772  ...  Leak112
        3            1.653696  ...  Leak112
        4            2.200430  ...  Leak112
        ...               ...  ...      ...
        3210      1617.966000  ...  Leak112
        3211      1618.495000  ...  Leak112
        3212      1619.025000  ...  Leak112
        3213      1619.554000  ...  Leak112
        3214      1620.083000  ...  Leak112
        <BLANKLINE>
        [3215 rows x 12 columns]

        ## A directory
        >>> directory = os.path.join(cwd, 'data', 'Leak')
        >>> RackioAI.load(directory)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...  Leak112
        1            0.502732  ...  Leak112
        2            1.232772  ...  Leak112
        3            1.653696  ...  Leak112
        4            2.200430  ...  Leak112
        ...               ...  ...      ...
        35397     1618.124000  ...  Leak120
        35398     1618.662000  ...  Leak120
        35399     1619.200000  ...  Leak120
        35400     1619.737000  ...  Leak120
        35401     1620.275000  ...  Leak120
        <BLANKLINE>
        [35402 rows x 12 columns]

        ```
        """
        (_, file_extension) = os.path.splitext(filename)
        specific_file = True

        if os.path.isfile(filename):

            (_, file_extension) = os.path.splitext(filename)

            if file_extension=='.tpl':

                tpl_file = True

        elif os.path.isdir(filename):

            specific_file = False

            if self.check_extension_files(filename, ext='.tpl'):

                tpl_file = True

        if tpl_file:

            self.tpl.read(filename, specific_file=specific_file)
            data = self.tpl.to('dataframe')

        else:

            raise KeyError('format file is not available to be loaded')

        return data

    @staticmethod
    def check_extension_files(root_directory, ext='.tpl'):
        """
        This is an utility method which you can check if in any directory exist files with *:param ext* extension

        **Parameters**

        * **:param root_directory:** (str) directory to look for files
        * **:param ext** (str) default='.tpl' extension file to look for

        **:return:**

        * **bool**: If True, exist *ext* in root_directory}

        ## Snippet code
        ```python
        >>> os.chdir('..')
        >>> cwd = os.getcwd()
        >>> directory = os.path.join(cwd, 'data', 'Leak')
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