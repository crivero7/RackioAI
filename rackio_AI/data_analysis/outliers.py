import numpy as np
import pandas as pd
from rackio_AI.utils.core import Utils
from easy_deco.progress_bar import ProgressBar
from random import uniform, choice
import scipy.stats as stats
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr



class Outliers:
    """
    In statistics, an outlier is a data point that differs significantly from other observations.
    An outlier may be due to variability in the measurement or it may indicate experimental error;
    the latter are sometimes excluded from the data set.[3] An outlier can cause serious problems 
    in statistical analyses.

    **Attributes**

    * **outliers:** (dict) Its keys are the dataframe columns with outliers. Keys:
        * **column_name:** (dict) Contains the following keys:
            * **locs:** (list) locations where were the outliers added
            * **values:** (list) Outliers values added
    * **detected:** (dict)
        * **column_name:** (dict) Contains the following keys:
            * **locs:** (list) locations where were the outliers added
            * **values:** (list) Outliers values added
            * **performance:** (float)


    """
    _instances = list()
        
    def __init__(self):

        self.outliers = dict()
        self.detected = dict()
        Outliers._instances.append(self)

    def add(
        self, 
        df: pd.DataFrame, 
        percent: float=5, 
        method: str="tf", 
        cols: list=None
        ):
        """
        Creates outliers values in a dataframe based on a given method

        **Parameters**

        * **:param df:** (pandas.DataFrame) Data to add outlier
        * **:param percent:** (float) outliers percent
        * **:param method:** (str) custom function name to calculate outlier
            * "tf": tukey-fence method
        * **:param cols:** (list) column names to add outliers, default None
            * If "None" outliers will be added to all columns

        **returns**

        * **df:** (pandas.DataFrame) Data with outliers added

        ______
        ### **Snippet code**

        ```python
        >>> import matplotlib.pyplot as plt
        >>> from rackio_AI import Outliers
        >>> df = pd.DataFrame(np.random.randn(100,2), columns=["a", "b"])
        >>> out = Outliers()
        >>> df = out.add(df)
        >>> ax = plt.plot(df["a"], '-r', df["b"], '-b', out.outliers["a"]["locs"], out.outliers["a"]["values"], 'rD', out.outliers["b"]["locs"], out.outliers["b"]["values"], 'bD')
        >>> ax = plt.legend(["a", "b", "a outliers", "b outliers"])
        >>> plt.show()

        ```
        ![Add Outlier](../img/add_outliers.png)

        """
        options = {
            "percent": percent,
            "method": method,
        }

        self._df_ = df.copy()

        if not cols:

            cols = Utils.get_column_names(df)

        self.__first_step_add(cols, **options)

        df = self._df_

        return df

    @ProgressBar(desc="Adding outliers...", unit="rows")
    def __first_step_add(self, column, **kwargs):
        """
        Documentation here
        """
        percent = kwargs["percent"]
        kwargs['col'] = column
        _subset_ = self._df_[column].sample(frac=percent / 100)
        self._cols_ = column
        self._locs_ = list(_subset_.index)
        self._values_ = list()
        self.__second_step_add(self._locs_, **kwargs)

        new_column = pd.Series(self._values_, name=self._cols_, index=self._locs_)
        self._df_.update(new_column)
        self.outliers[self._cols_] = {
            "locs": self._locs_,
            "values": self._values_
        }

        return

    @ProgressBar(desc="Adding outliers...", unit="columns", activate=False)
    def __second_step_add(self, index, **kwargs):
        """
        Documentation here
        """
        method = kwargs["method"]
        col = kwargs['col']
        window = self._df_[col].loc[index - 3: index + 3].values
        
        if method.lower() == "tf": # tukey fence method
            
            value = self.tukey_fence(window)

        self._values_.append(value)

        return

    def tukey_fence(
        self, 
        subset: np.ndarray,
        k_min:float=2, 
        k_max: float=5,
        q_min: float=0.25,
        q_max: float=0.75
        )->float:
        """
        A nonparametric outlier detection method. It is calculated by creating a 'fence' boundary a distance of
        k values * IQR beyond the 1st and 3rd quartiles. Any data beyond these fences are considered to be
        outliers.

        Outliers are values below q_min-k(q_max - q_min) or above q_max + k(q_max - q_min)

        **Parameters**

        * **:param subset:** (np.ndarray) values to calculate outlier based on interquartile
        * **:param k_min:** (float) lower boundary for tukey fence
        * **:param k_max:** (float) upper boundary for tukey fence
        * **:param q_min:** (float) between [0 - 1] lower quartile
        * **:param q_max:** (float) between [0 - 1] upper quartile

        **returns**

        * **value** (float) outlier value

        """
        q_min, q_max, iqr = self.iqr(subset, q_min, q_max)

        lower, low = q_min - k_max * iqr, q_min - k_min * iqr
        high, higher = q_max + k_min * iqr, q_max + k_max * iqr

        s = choice([0, 1])
        val = 0 if s else 1

        return s * uniform(lower, low) + val * uniform(high, higher)

    def iqr(
        self, 
        subset: np.ndarray, 
        q_min: float=0.25, 
        q_max: float=0.75,
        )-> tuple:
        """
        A nonparametric outlier detection method. It is calculated by creating a 'fence' boundary a distance of
        k values * IQR

        **Parameters**

        * **:param subset:** (np.ndarray) values to calculate outlier based on interquartile
        * **:param q_min:** (float) lower quartile
        * **:param q_max:** (float) upper quartile

        **returns**

        * **iqr** (tuple) (q_min, q_max, iqr)
            * **q_min** lower quartile from a subset
            * **q_max** upper quartile form a subset
            * **iqr** interquartile

        """
        q_min = np.quantile(subset, q_min)
        q_max = np.quantile(subset, q_max)
        iqr = q_max - q_min

        return q_min, q_max, iqr

    def z_score(self, df: pd.DataFrame, threshold: float=3)->list:
        """
        Rejects outlier values based on z-score modified

        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param threshold:** (float)

        **returns**

        * **y:** (list)
        """
        ys = df.values.flatten()
        median_y = np.median(ys)
        MAD = np.median([np.abs(y - ys) for y in ys])
        z_scores = [0.6745 * (y - median_y) / MAD for y in ys]

        mask = np.where(np.abs(z_scores) > threshold)

        return ys[mask]

    def detect(
        self,
        df: pd.DataFrame,
        win_size: int=30,
        step: int=1,
        conf: float=0.95,
        cols: list=None
        )->pd.DataFrame:
        """
        Detects any outliers values if exists in dataframe. If exists these outliers values 
        will be imputed.

        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param win_size:** (int)
        * **:param step:** (int)
        * **:param conf:** (float)
        * **:param cols:** (list)

        **returns**

        * **df:** (pandas.DataFrame)

        ______
        ### **Snippet code**

        ```python
        >>> import matplotlib.pyplot as plt
        >>> from rackio_AI import Outliers
        >>> df = pd.DataFrame(np.random.randn(1000,2), columns=["a", "b"])
        >>> out = Outliers()
        >>> df = out.add(df, percent=1)
        >>> df_imputed = out.detect(df, win_size=30)
        >>> ax = plt.plot(df["a"], '-r', df["b"], '-b', out.outliers["a"]["locs"], out.outliers["a"]["values"], 'rD', out.outliers["b"]["locs"], out.outliers["b"]["values"], 'bo', out.detected["a"]["locs"], out.detected["a"]["values"], 'kD', out.detected["b"]["locs"], out.detected["b"]["values"], 'ko')
        >>> ax = plt.legend(["a", "b", "a outliers", "b outliers", "a dectected", "b detected"])
        >>> plt.show()

        ```
        ![Detect Outlier](../img/impute_outliers.png)
    
        """
        self._df_ = df.copy()

        if not cols:

            cols = Utils.get_column_names(self._df_)

        options = {
            "win_size": win_size,
            "step": step
        }

        self._serie_list_ = Utils().get_windows(self._df_, win_size, step=step)

        self.__first_step_detect(cols, **options)

        df = self._df_

        return df

    @ProgressBar(desc="Detecting outliers...", unit="columns", activate=False)
    def __first_step_detect(self, col, **kwargs):
        """
        Documentation here
        """
        win_size = kwargs['win_size']
        step = kwargs['step']
        kwargs['col'] = col

        # self._serie_list_ = Utils().get_windows(self._df_, win_size, step=step)
        
        self._start_ = 0
        self._locs_ = list()
        self._values_ = list()
        self.__detect(self._serie_list_, **kwargs)
        self.detected.update({
            col: {
                "locs": self._locs_,
                "values": self._values_
                }
            })
        
        return

    @ProgressBar(desc="Detecting outliers...", unit="windows")
    def __detect(self, window, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of detect method

        **Parameters**

        * **:param window:** (list)

        **returns**

        None
        """
        col = kwargs['col']
        window = window.loc[:, col]
        likely = self.z_score(window)
        
        if likely.size > 0:
           
            likely = likely[0]
            loc = window[window == likely].index[0]

            if not loc in self._locs_:
                
                estimated = np.array([])
                
                if self._start_ < len(self._serie_list_) - 4:
                    
                    if self.check(likely, self._serie_list_[self._start_ + 1:self._start_ + 3], col):
                        
                        estimated = self.impute(likely, window)
                
                else:
                    
                    estimated = self.impute(likely, window)
                
                if estimated.size > 0:

                    self._locs_.append(loc)
                    self._values_.append(estimated)
                    self._df_[loc, col] = estimated
        
        self._start_ += 1

        return

    def check(self, value: float, subsets: list, col: str)->bool:
        """
        Documentation here
        """
        self._count_ = 0
        self._value_ = value
        self._status_outlier_ = False

        options = {
            'col': col
        }

        self.__check(subsets, **options)

        if self._count_ == 2:

            self._status_outlier_ = True

        return self._status_outlier_
    
    @ProgressBar(desc="Checking outlier detected...", unit="outlier", activate=False)
    def __check(self, subset, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of *check* method

        **Parameters**

        * **:param column_name:** (list)

        **returns**

        None
        """
        col = kwargs['col']
        subset = subset.loc[:, col]
        new_values = self.z_score(subset)

        if new_values.size > 0:

            if self._value_ in new_values:

                self._count_ += 1

        return

    def impute(
        self, 
        value: float, 
        sample: pd.Series,
        conf: float=0.95
        )->float:
        """
        Imputes outlier values using Auto Regressive method with two lags

        **Parameters**

        * **:param value:** (float)
        * **:param sample:** (pd.Series)
        * **:param conf:** (float)

        **returns**

        * **value:** (float)
        
        """
        qq = 1 - (1 - conf) / 2
        sample = sample.copy()
        sample.reset_index(drop=True, inplace=True)

        loc = np.where(np.asanyarray(~np.isnan(sample[sample == value])))[0][0]
        sample.iloc[loc, :] = np.nan
        sample.fillna(sample.median(), inplace=True)

        model = AutoReg(sample.values, lags=2, trend='n').fit()
        ss = np.std(model.resid)

        predictions = model.predict(start=0, end=len(sample) + 1)

        percent = stats.t.ppf(q=qq, df=len(sample) - 1)
        max_lim = predictions[loc] + percent * ss * np.sqrt(1 + 1 / len(sample))
        min_lim = predictions[loc] - percent * ss * np.sqrt(1 + 1 / len(sample))

        if Utils.is_between(min_lim, value, max_lim):

            return np.array([])

        elif Utils.is_between(min_lim, predictions[loc], max_lim):

            return predictions[loc]

        else:

            return sample.median()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
