import pandas as pd
from easy_deco.progress_bar import ProgressBar
import pickle
import rackio_AI


class PKL:
    """
    Documentation here
    """

    def read(self, pathname: str, **kwargs):
        """
        Documentation here
        """  
        self._df_ = list()
        self.__read(pathname, **kwargs)
        df = pd.concat(self._df_)
            
        return df

    @ProgressBar(desc="Reading .pkl files...", unit="file")
    def __read(self, pathname, **csv_options):
        """
        Read a comma-separated-values (csv) file into DataFrame.

        Also supports optionally iterating or breaking of the file into chunks.

        **Parameters**

        Same like read method
        """
        with open(pathname, 'rb') as f:
                
            self._df_.append(pickle.load(f))

        return
    

if __name__ == "__main__":
    import doctest

    doctest.testmod()