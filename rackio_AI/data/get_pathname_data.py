import os

def get_directory(folder_name):
    """

    """
    path_name = os.path.abspath(os.path.join('data', folder_name))

    return path_name