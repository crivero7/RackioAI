import pandas as pd
from easy_deco.progress_bar import ProgressBar
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr
import pickle
from random import shuffle


@set_to_methods(del_temp_attr)
class PKL:
    """
    This format allows to you read faster a DataFrame saved in pkl format
    """
    _instances = list()

    def __init__(self):

        super(PKL, self).__init__()

    def read(self, pathname: str, **kwargs):
        """
        Read a DataFrame saved with RackioAI's save method as a pkl file

        **Parameters**

        * **:param pathname:** (str) Filename or directory 

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
        >>> filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
        >>> df = RackioAI.load(filename)
        >>> print(df.head())
        tag      TIME_SERIES PT_SECTION_BRANCH_TUBERIA_PIPE_Pipe60_NR_1  ... CONTR_CONTROLLER_CONTROL_FUGA     file
        variable                                               Pressure  ...             Controller_output filename
        unit               S                                         PA  ...                                   .tpl
        0           0.000000                                   568097.3  ...                           0.0   Leak01
        1           0.502732                                   568098.2  ...                           0.0   Leak01
        2           1.232772                                   568783.2  ...                           0.0   Leak01
        3           1.653696                                   569367.3  ...                           0.0   Leak01
        4           2.200430                                   569933.5  ...                           0.0   Leak01
        <BLANKLINE>
        [5 rows x 12 columns]

        ```
        """  
        self._df_ = list()
        self.__read(pathname, **kwargs)
        if 'shuffle' in kwargs:
            _shuffle = kwargs['shuffle']
            if _shuffle:
                shuffle(self._df_)
            
        df = pd.concat(self._df_)
            
        return df

    @ProgressBar(desc="Reading .pkl files...", unit="file")
    def __read(self, pathname, **pkl_options):
        """
        Read (pkl) file into DataFrame.
        """
        with open(pathname, 'rb') as f:
            _df = pickle.load(f)

        if 'remove_initial_points' in pkl_options:
            _rip = pkl_options['remove_initial_points']

            _df.drop(index=_df.iloc[0:_rip, :].index.tolist(), inplace=True)
        
        self._df_.append(_df)

        return
    

if __name__ == "__main__":
    # import doctest

    # doctest.testmod()
    import os
    from rackio_AI import RackioAI, get_directory
    filename = os.path.join(get_directory('Leak'), 'Leak01.tpl')
    df = RackioAI.load(filename)