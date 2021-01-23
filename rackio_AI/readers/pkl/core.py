import pandas as pd
from easy_deco.progress_bar import ProgressBar
import pickle
import rackio_AI


class PKL:
    """
    This format allows to you read faster a DataFrame saved in pkl format
    """

    def read(self, pathname: str, **kwargs):
        """
        Read a DataFrame saved with RackioAI's save method as a pkl file

        **Parameters**

        * **:param pathname:** (str) Filename or directory 
        """  
        self._df_ = list()
        self.__read(pathname, **kwargs)
        df = pd.concat(self._df_)

        delattr(self, "_df_")
            
        return df

    @ProgressBar(desc="Reading .pkl files...", unit="file")
    def __read(self, pathname, **pkl_options):
        """
        Read (pkl) file into DataFrame.
        """
        with open(pathname, 'rb') as f:
                
            self._df_.append(pickle.load(f))

        return
    

if __name__ == "__main__":
    import doctest

    doctest.testmod()