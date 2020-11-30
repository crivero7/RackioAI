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
        >>> from rackio_AI import RackioAI, get_directory
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)

        ```
        ## An especific file

        ```python
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> RackioAI.load(filename)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...   Leak01
        1            0.502732  ...   Leak01
        2            1.232772  ...   Leak01
        3            1.653696  ...   Leak01
        4            2.200430  ...   Leak01
        ...               ...  ...      ...
        3214      1618.327000  ...   Leak01
        3215      1618.849000  ...   Leak01
        3216      1619.370000  ...   Leak01
        3217      1619.892000  ...   Leak01
        3218      1620.413000  ...   Leak01
        <BLANKLINE>
        [3219 rows x 12 columns]

        ```

        ## A directory

        ```python
        >>> directory = os.path.join(get_directory('Leak'))
        >>> RackioAI.load(directory)
        tag       TIME_SERIES  ...     file
        variable               ... filename
        unit                S  ...     .tpl
        0            0.000000  ...   Leak01
        1            0.502732  ...   Leak01
        2            1.232772  ...   Leak01
        3            1.653696  ...   Leak01
        4            2.200430  ...   Leak01
        ...               ...  ...      ...
        9648      1617.966000  ...   Leak02
        9649      1618.495000  ...   Leak02
        9650      1619.025000  ...   Leak02
        9651      1619.554000  ...   Leak02
        9652      1620.083000  ...   Leak02
        <BLANKLINE>
        [9653 rows x 12 columns]

        ```
        """
        (_, file_extension) = os.path.splitext(filename)
        tpl_file = False
        filenames = self.check_extension_files(filename, ext='.tpl')

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
        >>> from rackio_AI import get_directory
        >>> directory = os.path.join(get_directory('Leak'))
        >>> Reader.check_extension_files(directory)
        True

        ```
        """
        files = [os.path.join(r, fn) for r, ds, fs in os.walk(root_directory) for fn in fs if fn.endswith(ext)]
        #files = [f for f in glob.glob(root_directory + "**/*{}".format(ext), recursive=True)]

        if files:

            return files

        else:

            return False


if __name__ == "__main__":
    import doctest

    doctest.testmod()
