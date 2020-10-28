import os
import rackio_AI

def get_directory(folder_name=None):
    """

    """
    if folder_name:

        path_name = os.path.join(rackio_AI.__file__.replace(os.path.join(os.path.sep,'__init__.py'),''),'data',folder_name)

    else:

        path_name = os.path.join(rackio_AI.__file__.replace(os.path.join(os.path.sep, '__init__.py'), ''), 'data')

    return path_name