import os
import glob
from rackio_AI.readers.tpl import TPL


class Reader:
    """
    ...Documentation here...
    """

    tpl = TPL()


    def __init__(self):
        """

        """
        pass

    def read(self, filename):
        """
        ...Documentation here...

        **Parameters**

        * **:param filename:**

        **:return:**

        """
        (_, file_extension) = os.path.splitext(filename)
        specific_file = True

        if os.path.isfile(filename):

            (_, file_extension) = os.path.splitext(filename)

            if file_extension=='.tpl':

                tpl_file = True

        elif os.path.isdir(filename):

            specific_file = False

            if self.extension_files(filename, ext='.tpl'):

                tpl_file = True

        if tpl_file:

            self.tpl.read(filename, specific_file=specific_file)

        else:

            raise KeyError('format file is not available to be loaded')

    def extension_files(self, root_directory, ext='.tpl'):
        """
        ...Description here...
        * **:param root_directory:**

        **:return:**

        * **extension_files:** (list['str'])
        """

        files = [f for f in glob.glob(root_directory + "**/*{}".format(ext), recursive=True)]

        if files:

            return True

        else:

            return False