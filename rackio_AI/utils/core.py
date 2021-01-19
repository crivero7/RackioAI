from itertools import combinations as Combine
import json
import os


class Utils:
    """
    Documentation here
    """

    def __init__(self):
        """
        Documentation here
        """
        pass

    @staticmethod
    def check_default_kwargs(default_kw, kw):
        """
        Documentation here
        """
        kw = {key: kw[key] if key in kw.keys() else default_kw[key] for key in default_kw.keys()}
        
        return kw

    @staticmethod
    def get_column_names(df, **kwargs):
        """
        Documentation here
        pattern
        """
        default_kwargs = {
            "pattern": None
        }

        kw = Utils.check_default_kwargs(default_kwargs, kwargs)

        if not kw["pattern"]:
            
            return df.columns.to_list()

        else: 

            return

    @staticmethod
    def split_str(string: str, pattern: str, get_pos: int = 0):
        """
        Documentation here
        """
        return string.split(pattern)[get_pos]

    @staticmethod
    def get_combinations(columns=[], num=2):
        """
        Documentation here
        """
        return Combine(columns, num)

    @staticmethod
    def remove_row(df, loc):
        """
        Documentation here
        """
        df.pop(loc)
        
        return df

    @staticmethod
    def find_files(extension, path):
        """
        find all *:param extension:* files in *path*

        ___
        **Parameters**

        * **:param extension:** (str)
        * **:param path:** (str) root path

        **:return:**

        * **files:** (list['str'])

        ___

        ## Snippet code

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, Utils, get_directory

        ## An especific file
        >>> path = os.path.join(get_directory('Leak'))
        >>> Utils.find_files('.tpl', path)

        ```
        """
        result = list()

        for root, _, files in os.walk(path):

            for file in files:

                if file.endswith(extension):
                    result.append(os.path.join(root, file))

        return result

    @staticmethod
    def load_json(filename):
        """

        :return:
        """
        with open(filename, ) as f:

            return json.load(f)

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
        >>> files = Reader.check_extension_files(directory)

        ```
        """
        files = [os.path.join(r, fn) for r, ds, fs in os.walk(root_directory) for fn in fs if fn.endswith(ext)]

        if files:

            return files

        else:

            return False
            

class Rule:
    """
    Documentation here
    """

    def __init__(self):
        """
        Documentation here
        """
