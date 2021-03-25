from scipy.stats import kurtosis, skew
from rackio_AI.utils.utils_core import Utils
import pywt
import numpy as np
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


@set_to_methods(del_temp_attr)
class RackioAIFE:
    """
    Documentation here
    """
    _instances = list()

    def __init__(self):
       """Documentation here"""
       pass

    def kurt(
        self, 
        dataset, 
        axis: int=0, 
        fisher: bool=True, 
        bias: bool=True, 
        nan_policy: str='propagate'
        ):
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset

        Kurtosis is the fourth central moment divided by the square of the variance. If Fisher's definiton
        is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.

        If bias is False then the kurtosis is calculated using k statistics to eliminate bias coming from
        biased moment estimators

        **Parameters**

        * **dataset:** (2d array) Data for which the kurtosis is calculated
        * **axis:** (int or None) Axis along which the kurtosis is calculated. Default is 0. If None, compute
        over the whole array dataset.
        * **fisher:** (bool) If True, Fisher's definition is used (normal ==> 0.0). If False, Pearson's deifnition
        is used (normal ==> 3.0)
        * **bias:** (bool) If False, then the calculations are corrected for statistical bias.
        * **nan_policy:** ({'propagate', 'raise', 'omit'}) Defines how to handle when inputs contains nan. 'propagate' 
        returns nan, 'raise' throws an error, 'omit' performs the calculations ignoring nan values. Default is propagate.

        **returns**

        * **kurtosis** (array 1xcols_dataset) The kurtosis of values along an axis. If all values are equal, return -3 for Fisher's definition
        and 0 for Pearson's definition

        ## Snippet Code
        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> dataset = norm.rvs(size=1000, random_state=3)
        >>> feature_extraction.kurt(dataset)
        array([-0.06928694])

        ```

        ## Snippet Code
        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> dataset = norm.rvs(size=(1000,2), random_state=3)
        >>> feature_extraction.kurt(dataset)
        array([-0.00560946, -0.1115389 ])

        ```
        """
        dataset = self.__check_dataset_shape(dataset)
        _, cols = dataset.shape
        _kurt = np.array([kurtosis(
            dataset[:, col], 
            axis=axis, 
            fisher=fisher, 
            bias=bias, 
            nan_policy=nan_policy
            ) for col in range(cols)]
        )

        return _kurt

    def mean(
        self, 
        dataset,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue
        ):
        """
        Compute the arithmetic mean along the specified axis.
        Returns the average of the array elements.  The average is taken over
        the flattened array by default, otherwise over the specified axis.
        `float64` intermediate and return values are used for integer inputs.
        
        **Parameters**
        
        * **dataset:** (2d array_like) Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
        * **axis:** (None or int or tuple of ints, optional) Axis or axes along which the means are computed. 
        The default is to compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis or all the
        axes as before.
        * **dtype:** (data-type, optional) Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the input dtype.
        * **out:** (ndarray, optional) Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the expected output, but the type will be cast if
        necessary.
        * **keepdims:** (bool, optional) If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option, the result will broadcast correctly against the 
        input array. If the default value is passed, then `keepdims` will not be passed through to the `mean` method 
        of sub-classes of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement
        `keepdims` any exceptions will be raised.
        
        **Returns**
        
        * **m:** (ndarray, see dtype parameter above) If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> dataset = np.array([[1, 2], [3, 4]])
        >>> feature_extraction.mean(dataset)
        2.5
        >>> feature_extraction.mean(dataset, axis=0)
        array([2., 3.])
        >>> feature_extraction.mean(dataset, axis=1)
        array([1.5, 3.5])
        
        ```
        """
        dataset = self.__check_dataset_shape(dataset)
        _, cols = dataset.shape
        _mean = np.mean(dataset, axis=axis, dtype=dtype, out=dtype, keepdims=keepdims)

        return _mean

    def std(self, data):
        """Documentation here"""
        _std = [np.std(col, axis=0) for col in data]
        _std = np.concatenate(_std, axis=0)
        _std = _std.reshape((data.shape[0], data.shape[2]))
        return _std

    def skew(self, data):
        """Documentation here"""
        _skew = [skew(col, axis=0) for col in data]
        _skew = np.concatenate(_skew, axis=0)
        _skew = _skew.reshape((data.shape[0], data.shape[2]))
        return _skew

    def rms(self, data):
        """Documentation here"""
        _rms = [(np.sum(col ** 2, axis=0) / data.shape[0]) ** 0.5 for col in data]
        _rms = np.concatenate(_rms, axis=0)
        _rms = _rms.reshape((data.shape[0], data.shape[2]))
        return _rms

    def peak_2_valley(self, data):
        """Documentation here"""
        _peak_2_valley = [(np.max(col, axis=0)-np.min(col, axis=0)) / 2 for col in data]
        _peak_2_valley = np.concatenate(_peak_2_valley, axis=0)
        _peak_2_valley = _peak_2_valley.reshape((data.shape[0], data.shape[2]))
        return _peak_2_valley

    def peak(self, data):
        """Documentation here"""
        _peak = [np.max(col - col[0,:], axis=0) for col in data]
        _peak = np.concatenate(_peak, axis=0)
        _peak = _peak.reshape((data.shape[0], data.shape[2]))
        return _peak

    def crest_factor(self):
        """Documentation here"""
        peak = self.peak()
        rms = self.rms()
        return np.array([peak[i,:] / rms[i,:] for i in range(peak.shape[0])])

    def statistical(
        self, 
        data, 
        std=True, 
        mean=False,
        rms=False,
        kurt=False,
        skew=False,
        peak=False,
        peak_2_valley=False,
        crest_factor=False,
        concatenate=True
        ):
        """Documentation here"""
        result = list()
        features = {
            'std': std,
            'mean': mean,
            'rms': rms, 
            'kurt': kurt, 
            'skew': skew, 
            'peak': peak, 
            'peak_2_valley': peak_2_valley, 
            'crest_factor': crest_factor
        }

        for key, value in features.items():
            
            if value:
                feature = getattr(self, key)
                result.append(feature(data))

        if concatenate:

            result = np.concatenate(result, axis=1)

        return result

    def wavelets(self, waveletType='db2', lvl=3):
        """Documentation here"""
        waveletFeatures = list()
        lastCoeffs = 3
        for data in self.inputData:
            coeffs = pywt.wavedec(data, waveletType, level=lvl, axis=0)
            waveletFeatures.append(np.concatenate([sum(coeff ** 2, 0) for coeff in coeffs[0:lastCoeffs]],axis=0))
        return np.array(waveletFeatures)

    @staticmethod
    def __check_dataset_shape(dataset):
        """Documentation here"""

        if len(dataset.shape) == 1:
            dataset = np.atleast_2d(dataset)
            rows, cols = dataset.shape
            if cols > rows:
                dataset = dataset.reshape((-1,1))

        elif len(dataset.shape) == 3:

            raise TypeError("dataset shape must be 2d")

        return dataset


if __name__=='__main__':

    import doctest

    doctest.testmod()