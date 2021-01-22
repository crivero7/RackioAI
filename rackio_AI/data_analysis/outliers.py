import numpy as np
import pandas as pd
from rackio_AI import Utils
from easy-decor.progress_bar import ProgressBar
from random import uniform, choice


class Outliers:
    """
    Documentation here

    **Attributes**

    * **locs:** (list)
    * **value:** (list)

    """
    def __init__(self):
        """
        Documentation here
        """
        self.locs = list()
        self.values = list()

    def add(
        self, 
        df: pd.DataFrame, 
        percent: float=75, 
        fn: str="IQR", 
        cols: list=None
        ):
        """
        Creates outliers values in a dataframe based on a given function

        **Parameters**

        * **:param df:** (pandas.DataFrame) Data to add outlier
        * **:param percent:** (float) outliers percent
        * **:param fn:** (str) custom function name to calculate outlier
            * "IQR": interquartile
        * **:param cols:** (list) column names to add outliers, default None
            * If "None" outliers will be added to all columns

        **returns**

        * **df:** (pandas.DataFrame) Data with outliers added

        """
        options = {
            "percent": percent,
            "fn"= fn,
        }

        self._df_ = df

        if not cols:

            cols = Util.get_column_names(df)

        self.__first_step_add(cols, **options)

        return

    @ProgressBar(desc="Adding outliers...", unit="rows")
    def __first_step_add(self, column, **kwargs):
        """
        Documentation here
        """
        percent = kwargs["percent"]
        self._subset_ = self._df_[column].sample(frac=percent / 100, random_state=44)
        self.locs = list(_subset.index)

        self.__second_step_add(self.locs, **kwargs)

        return

    @ProgressBar(desc="Adding outliers...", unit="columns")
    def __second_step_add(self, index, **kwargs):
        """
        Documentation here
        """
        fn = kwargs["fn"]
        window = self._subset_.loc[index - 3: index + 3].values
        
        if fn.lower() == "iqr":
            
            value = self.__iqr(window)

        self.values.append(value)

        return


    def __iqr(self, subset: np.ndarray)-> float:
        """
        Calculates outlier based on interquartile of a dataframe

        **Parameters**

        * **:param subset:** (np.ndarray) values to calculate outlier based on interquartile
        
        **returns**

        * **value** (float) outlier value

        """
        q1 = np.quantile(subset, 0.25)
        q3 = np.quantile(subset, 0.75)
        iqr = q3 - q1

        lower, low = q1 - 5 * iqr, q1 - 2 * iqr
        high, higher = q3 + 2 * iqr, q3 + 5 * iqr

        s = choice([0, 1])
        val = 0 if s else 1

        return s * uniform(lower, low) + val * uniform(high, higher)

    def __z_score(self, df: pd.DataFrame, threshold: float=3)->list:
        """
        Rejects outlier values based on z-score modified

        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param threshold:** (float)

        **returns**

        * **y:** (list)
        """
        ys = df.to_numpy().flatten()
        median_y = np.median(ys)
        MAD = np.median([np.abs(y - ys) for y in ys])
        z_scores = [0.6745 * (y - median_y) / MAD for y in ys]

        mask = np.where(np.abs(z_scores) > threshold)

        return ys[mask]

    def detect(
        self,
        df: pd.DataFrame,
        winsize: int=30,
        step: int=1,
        conf: float=0.95):
        """
        documentation here
        """
        pass

    def __check(self):
        """
        Documentation here
        """
        pass

    def __fix(self):
        """
        Documentation here
        """
        pass
