from easy_deco.progress_bar import ProgressBar
from itertools import combinations as Combine
import json
import pickle
import math
import numpy as np
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
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)  
     
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
    def load_pickle(filename: str):
        """
        Accepts file object, parses the pickle object, populates a Python dictionary 
        with the data and returns it back to you.

        **Parameters**

        * **:param filename:** (str) pickle filename

        **returns**

        json file object parsed
        """
        with open(filename, 'rb') as f:

            return pickle.load(f)

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
        >>> from rackio_AI import get_directory, Utils
        >>> directory = os.path.join(get_directory('Leak'))
        >>> files = Utils.check_extension_files(directory)

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

    @staticmethod
    def round(
        value: float, 
        decimals: int=0,
        down: bool=True
        )->float:
        """
        Round down or up a value

        **Parameters**

        * **:param value:** (float) value to round
        * **:param decimals:** (int) decimals to round
        * **:param down:** (bool)
            * If *True* round down
            * Otherwise round up 

        **returns**

        * **value** (float) value rounded down

        ____

        ### **Snippet code**

        ```python
        >>> from rackio_AI import Utils
        >>> value = 12.3456
        >>> Utils.round(value, decimals=2)
        12.34
        >>> Utils.round(value, decimals=2, down=False)
        12.35

        ```
        """ 
        multiplier = 10 ** decimals

        if down:

            return math.floor(value * multiplier) / multiplier
        
        else:

            return math.ceil(value * multiplier) / multiplier

    @staticmethod
    def is_between(
        min_value: float, 
        value: float, 
        max_value: float
        )-> bool:
        """
        Check if a value is between a min and a  max value

        **Parameters**

        * **:param min_value:** (float) lower value
        * **:param value:** (float) value to check this conditional
        * **:param max_value:** (float) higher value

        **returns**

        **bool**

        ___

        ### Snippet code

        ```python 
        >>> from rackio_AI import Utils
        >>> Utils.is_between(1, 5.6, 10)
        True
        >>> Utils.is_between(3.9, 2, 10.5)
        False
        
        ```
        """

        return min(min_value, max_value) < value < max(min_value, max_value)

    def get_windows(
        self,
        df: pd.DataFrame,
        win_size: int,
        step: int
        ):
        """
        Creates sliding windows from a dataframe

        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param win_size:** (int) window size
        * **:param step:** (int) window sliding step

        **return**

        * **slide** (generator)

        ______
        ### **Snippet code**

        ```python

        ```
        """
        if step > win_size:

            raise ValueError("Step must be less than win_size")
        
        self._df_ = df
        self._start_ = -step
        self._windows_ = list()
        options = {
            'win_size': win_size,
            'step': step
        }
        _slice = range(0, len(df) - win_size + step, step)
        self.__get_windows(_slice, **options)
        
        return self._windows_

    @ProgressBar(desc="Creating windows...", unit="windows")
    def __get_windows(self, col, **kwargs):
        """
        Documentation here
        """
        win_size = kwargs['win_size']
        step = kwargs['step']
        self._start_ += step
        self._df_.iloc[self._start_: self._start_ + win_size, :]
        self._windows_.append(self._df_.iloc[self._start_: self._start_ + win_size, :])

        return

    @staticmethod 
    def check_dataset_shape(dataset):
        """
        Documentation here
        """
        if len(dataset.shape) == 1:
            dataset = np.atleast_2d(dataset)
            rows, cols = dataset.shape
            if cols > rows:
                dataset = dataset.reshape((-1,1))

        elif len(dataset.shape) == 3:

            raise TypeError("dataset shape must be 2d")

        return dataset

if __name__ == "__main__":
    import doctest

    doctest.testmod()

