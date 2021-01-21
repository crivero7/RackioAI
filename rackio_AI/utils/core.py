from itertools import combinations as Combine
import json
import os
import pandas as pd


class Utils:
    """
    Encapsulates only static methods useful in any class
    """

    @staticmethod
    def check_default_kwargs(default_kw: dict, kw: dict) -> dict:
        """
        Given any keyword arguments *kw*, check if their keys are in default keyword arguments *default_kw*

        * If any key in *kw* is in *default_kw* replace kw's key value in default_kw's key
        * Otherwise *default_kw* keeps it key value.

        **Parameters**

        * **:param defult_kw:** (dict) Default keyword arguments.
        * **:param kw:** (dict) Keyword arguments to check.

        **returns**

        * **kw:** (dict) Keyword arguments checked
        """
        kw = {key: kw[key] if key in kw.keys() else default_kw[key] for key in default_kw.keys()}
        
        return kw

    @staticmethod
    def get_column_names(df: pd.DataFrame) -> list:
        """
        Get columns names given a dataframe

        **Parameters**

        * **:param df:** (pd.DataFrame)

        **returns**

        **column_names** (list)
        """            
        return df.columns.to_list()

    @staticmethod
    def load_json(filename: str):
        """
        Accepts file object, parses the JSON data, populates a Python dictionary 
        with the data and returns it back to you.

        **Parameters**

        * **:param filename:** (str) json filename

        **returns**

        json file object parsed
        """
        with open(filename, ) as f:

            return json.load(f)

    @staticmethod
    def check_extension_files(root_directory: str, ext: str='.tpl'):
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
        >>> files = Reader.check_extension_files(directory)

        ```
        """
        files = [os.path.join(r, fn) for r, ds, fs in os.walk(root_directory) for fn in fs if fn.endswith(ext)]

        if files:

            return files

        else:

            return False

    @staticmethod
    def split_str(string: str, pattern: str, get_pos: int = 0) -> str:
        """
        Split string given a *pattern* and get the position *get_pos*

        **Parameters**

        * **:param string:** (str) String to split
        * **:param pattern:** (str) String to look for in *String* to split
        * **:param get_pos:** (int) Get string in the position get_pos after split string

        **returns**

        * **string**
        """
        return string.split(pattern)[get_pos]

    @staticmethod
    def check_path(pathname: str, ext: str=".tpl") -> tuple:
        """
        Checks if a pathname is a directory or a file

        **Parameters**

        * **:param pathname:** (str)
        * **:param ext:** (str) file extension to look for

        **returns**

        * **(filenames, ext):** (tuple)

        """
        (pathname, file_ext) = os.path.splitext(pathname)
        
        if not file_ext:

            pathname = Utils.check_extension_files(pathname, ext=ext)
            file_ext = ext
            
            if not pathname:

                raise FileNotFoundError("File not found in {} directory with {} extension".format(pathname, ext))  

        else:
             
            pathname = [pathname + file_ext]      

        return pathname, file_ext
