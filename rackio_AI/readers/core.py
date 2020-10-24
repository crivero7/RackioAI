import os
import glob
from rackio_AI.readers.tpl import TPL



class Reader:

    tpl = TPL()


    def __init__(self):
        """

        :param filename:
        """
        pass

    def read(self, filename):
        """

        """
        (_, file_extension) = os.path.splitext(filename)

        if os.path.isfile(filename):

            (_, file_extension) = os.path.splitext(filename)
            reader = getattr(self, file_extension.replace('.',''))
            reader.read(filename)

        elif os.path.isdir(filename):

            if self.extension_files(filename, ext='.tpl'):

                self.tpl.read(filename, specific_file=False)

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