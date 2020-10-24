import os
from rackio_AI.readers.tpl import TPL



class Reader:

    tpl = TPL()


    def __init__(self):
        """

        :param filename:
        """

        # if os.path.isfile(filename):
        #
        #     (path_filename , file_extension) = os.path.splitext(filename)
        #
        #     self.filename = filename
        #     self.path_filename = path_filename
        #
        # elif os.path.isdir(filename):
        #
        #     self.path_filename = filename
        #     self.filename = None
        #     self.doc = list()

    def read(self, filename):
        """

        """
        (_, file_extension) = os.path.splitext(filename)

        if os.path.isfile(filename):

            (_, file_extension) = os.path.splitext(filename)
            reader = getattr(self, file_extension.replace('.',''))
            reader.read(filename)

        elif os.path.isdir(filename):

            extension_files = self.get_extension_files()

            if '.tpl' in extension_files:

                self.tpl.read(filename)
