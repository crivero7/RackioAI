import numpy as np
import pandas as pd
from easy_deco.progress_bar import ProgressBar
from rackio_AI.utils.utils_core import Utils
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


class Noise:
    """
    Encapsulates method to work with noise
    """
    _instances = list()

    def __init__(self):

        Noise._instances.append(self)

    def add(
        self, 
        df: pd.DataFrame,
        win_size: int=30,
        method: str="rhinehardt",
        cols: list=None,
        std_factor: float=0.001
        )-> pd.DataFrame:
        """
        Add gaussian noise over subsequence windows based on some method

        **Parameters**

        * **:param df:** (pandas.DataFrame)
        * **:param win_size:** (int) window size to apply gaussian noise
        * **:param method:** (str) method to base gaussian noise
            * *rhinehardt* or *rh*
        * **:param cols:** (list) column names to add gaussian noise.

        **returns**

        * **df** (pandas.DataFrame) noise added

        ______
        ### **Snippet code

        ```python
        >>> import matplotlib.pyplot as plt
        >>> from rackio_AI import Noise 
        >>> df = pd.DataFrame(np.random.randn(100,2), columns=["a", "b"])
        >>> noise = Noise()
        >>> df_noisy = noise.add(df, win_size=10)
        >>> ax = plt.plot(df.index, df["a"], '-r', df.index, df["b"], '-b', df_noisy.index, df_noisy["a"], '--r', df_noisy.index, df_noisy["b"], '--b')
        >>> ax = plt.legend(["a", "b", "noisy a", "noisy b"])
        >>> plt.show()

        ```
        ![Add rhinehardt noise](../img/rhinehardt_noise.png)
        """
        options = {
            'win_size': win_size,
            'method': method,
            'std_factor': std_factor
        }
        self._df_ = df.copy()
        if not cols:

            cols = Utils.get_column_names(self._df_)

        self.__first_step_add(cols, **options)

        df = self._df_

        return df

    @ProgressBar(desc="Adding gaussian noise...", unit="columns")
    def __first_step_add(self, col, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of *add noise* method

        **Parameters**

        * **:param column_name:** (list)

        **returns**

        None
        """
        win_size = kwargs['win_size']
        windows_number = self._df_.shape[0] // win_size + 1
        windows = np.array_split(self._df_.loc[:, col], windows_number, axis=0)
        self._noise_ = list()

        self.__last_step_add(windows, **kwargs)

        self._df_.loc[:, col] = self._noise_

        return

    @ProgressBar(desc="Adding gaussian noise...", unit="windows")
    def __last_step_add(self, window, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of *add noise* method

        **Parameters**

        * **:param column_name:** (list)

        **returns**

        None
        """
        method = kwargs['method']

        if method.lower() in ["rhinehardt", "rh"]:
            std_factor = kwargs['std_factor']
            self._noise_.extend(self.rhinehardt(window, std_factor=std_factor))

        return

    def rhinehardt(self, x: pd.DataFrame, std_factor: float=1)->np.ndarray:
        """
        Add noise to variable x based on Box-Muller transform

        **Parameters**

        * **:param x:** (pandas.DataFrame)
        """
        x = x.values
        x = x.flatten()
        rng = np.random.RandomState(seed=42)
        r1, r2 = rng.uniform(size=len(x)), rng.uniform(size=len(x))
        xmean = np.mean(x)
        s = np.sqrt(np.sum((xmean - x)**2) / (len(x) - 1))

        if s <= 1:

            s = std_factor * xmean

        d = s * np.sqrt(-2 * np.log(r1)) * np.sin(2 * np.pi * r2)

        return x + d


if __name__ == "__main__":
    import doctest

    doctest.testmod()
